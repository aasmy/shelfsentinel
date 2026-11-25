import json
import os
import time

import cv2
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

from shelfsentinel.llm.provider_status import RateLimitError, DailyLimitError, TransientError

RPM_MSG = "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"
RPD_MSG = "GenerateRequestsPerDayPerProjectPerModel-FreeTier"
load_dotenv()  # reads GEMINI_API_KEY from .env


class Gemini:
    def __init__(self, model=None):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is missing, add it to .env")

        # json mode only works on v1beta, the sdk defaults to v1
        self.client = genai.Client(
           api_key=api_key,
           http_options=types.HttpOptions(
               api_version="v1beta",
               retry_options=types.HttpRetryOptions(
                   attempts=1
               )
           )
        )
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

    def ask(self, prompt):
        response = self.client.models.generate_content(model=self.model, contents=prompt)
        return response.text

    def ask_with_images(self, prompt, frames):
        # frames are opencv images (numpy arrays), we send them as jpg
        parts = []
        for frame in frames:
            ok, jpg = cv2.imencode(".jpg", frame)
            if ok:
                parts.append(types.Part.from_bytes(data=jpg.tobytes(), mime_type="image/jpeg"))
        parts.append(prompt)

        config = types.GenerateContentConfig(response_mime_type="application/json")  # ask for json back
        try:
            response = self.client.models.generate_content(model=self.model, contents=parts, config=config)
        except errors.APIError as e:
            if e.code == 429:
                body = e.details if isinstance(e.details, dict) else {}
                err = body.get("error", body)  # handle both response shapes
                daily, retry_after = False, None
                for detail in err.get("details", []):
                    kind = detail.get("@type", "")
                    if "RetryInfo" in kind and detail.get("retryDelay"):
                        retry_after = float(detail["retryDelay"].rstrip("s"))
                    elif "QuotaFailure" in kind:
                        for v in detail.get("violations", []):
                            if "PerDay" in v.get("quotaId", ""):
                                print(v["quotaId"])
                                daily = True
                if daily:
                    raise DailyLimitError(e.message) from e
                raise RateLimitError(e.message,
                                     retry_after=(retry_after or 60) + 1) from e
                raise TransientError(e.message) from e
            elif e.code == 503:
                raise TransientError(msg=e.message)
            else: raise TransientError(msg=e.message)
            if not response.text:
                raise TransientError("empty response (possibly blocked)")
        return json.loads(response.text)
