import time

class ProviderError(Exception):
    pass


class DailyLimitError(ProviderError):
    """Quota is gone for the day. Disable the provider for the rest of the run."""


class RateLimitError(ProviderError):
    """Per-minute style limit. Provider is fine again after retry_after seconds."""
    def __init__(self, msg="", retry_after=60):
        super().__init__(msg)
        self.retry_after = retry_after


class TransientError(ProviderError):
    """503 / timeout / bad JSON etc. Short backoff, then try again."""
    def __init__(self, msg="", retry_after=None):
        super().__init__(msg)
        self.retry_after = retry_after  # None -> manager uses exponential backoff



class ProviderStatus:
    def __init__(self, name, client, base_backoff=5, max_backoff=120):
        self.name = name
        self.client = client
        self.available_at = 0.0
        self.disabled = False
        self.disable_reason = None
        self.consecutive_failures = 0
        self.base_backoff = base_backoff
        self.max_backoff = max_backoff

    def is_available(self):
        return not self.disabled and time.monotonic() >= self.available_at

    def seconds_until_available(self):
        return max(0.0, self.available_at - time.monotonic())

    def cooldown(self, seconds):
        self.available_at = max(self.available_at, time.monotonic() + seconds)

    def disable(self, reason):
        self.disabled = True
        self.disable_reason = reason

    def record_success(self):
        self.consecutive_failures = 0

    def record_failure(self):
        """Returns an exponential backoff (5s, 10s, 20s ... capped)."""
        self.consecutive_failures += 1
        return min(self.max_backoff,
                   self.base_backoff * 2 ** (self.consecutive_failures - 1))
