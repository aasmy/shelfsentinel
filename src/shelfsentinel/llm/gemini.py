import os

from dotenv import load_dotenv
from google import genai

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
