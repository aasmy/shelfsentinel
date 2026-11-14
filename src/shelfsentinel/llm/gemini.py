import json
import os
import time

import cv2
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

load_dotenv()  # reads GEMINI_API_KEY from .env


class DailyLimitError(Exception):
    pass


class Gemini:
    def __init__(self, model=None):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is missing, add it to .env")

        # json mode only works on v1beta, the sdk defaults to v1
        self.client = genai.Client(api_key=api_key, http_options=types.HttpOptions(api_version="v1beta"))
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

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
            if e.code == 429 and "per day" in str(e):
                raise DailyLimitError(str(e))  # no point retrying, quota resets tomorrow
            if e.code not in (429, 503):
                raise
            # 429 = per minute limit, 503 = model is busy. both go away if we wait a bit
            print(f"  gemini said {e.code}, waiting 60s and trying again...")
            time.sleep(60)
            response = self.client.models.generate_content(model=self.model, contents=parts, config=config)
        return json.loads(response.text)
