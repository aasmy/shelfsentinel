import time
from shelfsentinel.llm.provider_status import DailyLimitError, ProviderError, RateLimitError, TransientError

class AllProvidersExhausted(ProviderError):
    """Every provider is disabled for this run."""



class ProviderManager:
    def __init__(self, providers, max_attempts=None, log=print):
        """
        providers: list of ProviderStatus, in priority order (best/cheapest first).
        max_attempts: max consecutive failures with no progress before giving up
                      (default: 3 * number of providers).
        """
        self.providers = providers
        self.max_attempts = max_attempts or 3 * len(providers)
        self.log = log

    # -- selection ---------------------------------------------------------

    def _pick(self):
        """Return (provider, wait_seconds). provider is None if all are disabled."""
        live = [p for p in self.providers if not p.disabled]
        if not live:
            return None, 0.0
        for p in live:  # first available in priority order
            if p.is_available():
                return p, 0.0
        soonest = min(live, key=lambda p: p.available_at)
        return soonest, soonest.seconds_until_available()

    # -- error handling ----------------------------------------------------

    def _handle_error(self, provider, e):
        if isinstance(e, DailyLimitError):
            provider.disable(f"daily limit: {e}")
            self.log(f"  {provider.name} hit its daily limit, disabled for this run")
        elif isinstance(e, RateLimitError):
            provider.cooldown(e.retry_after)
            self.log(f"  {provider.name} rate limited, cooling down {e.retry_after:.0f}s")
        elif isinstance(e, TransientError):
            secs = e.retry_after if e.retry_after is not None else provider.record_failure()
            provider.cooldown(secs)
            self.log(f"  {provider.name} failed ({e}), cooling down {secs:.0f}s")
        else:  # ValueError from json.loads -> model returned junk
            secs = provider.record_failure()
            provider.cooldown(secs)
            self.log(f"  {provider.name} returned invalid JSON, cooling down {secs:.0f}s")

    # -- main entry point --------------------------------------------------

    def process_frames(self, prompt, frames, batch_size=1, on_result=None):
        """
        Work through `frames` in order and return one result per batch.

        A provider keeps going until it errors. The failed batch and everything
        after it is then handed to the next available provider.

        batch_size: frames per request (1 = one request per frame).
        on_result:  optional callback(batch_index, result), called as each batch
                    succeeds so you can save progress as you go.

        On failure, the raised exception carries `.partial_results`.
        """
        batches = [frames[i:i + batch_size] for i in range(0, len(frames), batch_size)]
        results = []
        idx = 0
        stalled = 0  # consecutive failures with no progress

        while idx < len(batches):
            provider, wait = self._pick()

            if provider is None:
                reasons = ", ".join(f"{p.name}: {p.disable_reason}" for p in self.providers)
                err = AllProvidersExhausted(
                    f"all providers disabled at batch {idx + 1}/{len(batches)} ({reasons})")
                err.partial_results = results
                raise err

            if wait > 0:
                self.log(f"  all providers cooling down; waiting {wait:.0f}s for {provider.name}")
                time.sleep(wait)

            self.log(f"  {provider.name} taking batch {idx + 1}/{len(batches)}")

            # this provider keeps working until it hits an error
            while idx < len(batches):
                try:
                    result = provider.client.ask_with_images(prompt, batches[idx])
                except (ProviderError, ValueError) as e:
                    self._handle_error(provider, e)
                    stalled += 1
                    break  # hand the failed batch + the rest to the next provider

                provider.record_success()
                results.append(result)
                if on_result:
                    on_result(idx, result)
                idx += 1
                stalled = 0

            if stalled >= self.max_attempts:
                err = ProviderError(f"no progress after {stalled} consecutive failures")
                err.partial_results = results
                raise err

        return results

    def ask_with_images(self, prompt, frames):
        """Single request with all frames, with failover (the old behaviour)."""
        return self.process_frames(prompt, frames, batch_size=len(frames))[0]