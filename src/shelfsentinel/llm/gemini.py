import json
import os

import cv2
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()  # reads GEMINI_API_KEY from .env


class Gemini:
    def __init__(self, model=None):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is missing, add it to .env")

        self.client = genai.Client(api_key=api_key)
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

        response = self.client.models.generate_content(
            model=self.model,
            contents=parts,
            config=types.GenerateContentConfig(response_mime_type="application/json"),  # ask for json back
        )
        return json.loads(response.text)
