import base64
import json
import os

import cv2
from dotenv import load_dotenv
load_dotenv()
import groq

from shelfsentinel.llm.provider_status import RateLimitError, DailyLimitError, TransientError

class GroqApi:
    def __init__(self, model=None):
        self.client = groq.Groq(
            api_key=os.environ.get("GROQ_API_KEY")
        )
        self.model = model or os.getenv("GROQ_MODEL")

    def ask_with_images(self, prompt, frames):
        parts = [
            {
                "type": "text",
                "text": prompt
            }
        ]

        for frame in frames:
            ok, jpg = cv2.imencode(".jpg", frame)

            if ok:
                encoded_image = base64.b64encode(
                    jpg.tobytes()
                ).decode("utf-8")

                parts.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encoded_image}"
                    }
                })

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a video analysis assistant."
                    },
                    {
                        "role": "user",
                        "content": parts
                    }
                ],
                response_format={"type": "json_schema",
                                 "json_schema": {
                                     "name": "video_analysis",
                                     "strict": True,
                                     "schema": {
                                         "type": "object",
                                         "properties": {
                                             "hands": {"type": "string"},
                                              "item_taken": {
                                                "type": "string",
                                                "enum": ["true", "false"]
                                              },
                                              "item_hidden": {
                                                "type": "string",
                                                "enum": ["true", "false"]
                                              },
                                              "clear_view": {
                                                "type": "string",
                                                "enum": ["true", "false"]
                                              }
                                         },
                                         "required": ["hands", "item_taken",
                                                      "item_hidden",
                                                      "clear_view"],
                                         "additionalProperties": False
                                     }

                                 }
                                },
                max_completion_tokens=2048
            )

        except groq.RateLimitError as e:
            if e.status_code == 429:
                headers = e.response.headers
                if int(headers.get("x-ratelimit-remaining-requests", 1)) > 0:
                    raise RateLimitError(msg=e.message, retry_after=int(headers.get("retry-after", 30)))
                else: raise DailyLimitError
            else:
                raise TransientError(e.message)
        except (groq.APIStatusError, groq.APIConnectionError) as e:
            raise TransientError(str(e)) from e
        except Exception as e:
            raise
        return json.loads(
            response.choices[0].message.content or "{}"
        )