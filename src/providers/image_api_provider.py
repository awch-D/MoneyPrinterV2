import base64
from dataclasses import dataclass

import requests

from config import (
    get_nanobanana2_api_base_url,
    get_nanobanana2_api_key,
    get_nanobanana2_aspect_ratio,
    get_nanobanana2_model,
)


@dataclass
class ImageApiProvider:
    timeout_seconds: int = 300

    def generate_image_bytes(self, prompt: str) -> bytes:
        api_key = get_nanobanana2_api_key()
        if not api_key:
            raise RuntimeError("Missing config: nanobanana2_api_key or GEMINI_API_KEY")

        endpoint = (
            f"{get_nanobanana2_api_base_url()}/models/{get_nanobanana2_model()}:generateContent"
        )
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "responseModalities": ["IMAGE"],
                "imageConfig": {"aspectRatio": get_nanobanana2_aspect_ratio()},
            },
        }

        response = requests.post(
            endpoint,
            headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
            json=payload,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        body = response.json()

        candidates = body.get("candidates", [])
        for candidate in candidates:
            content = candidate.get("content", {})
            for part in content.get("parts", []):
                inline_data = part.get("inlineData") or part.get("inline_data")
                if not inline_data:
                    continue
                data = inline_data.get("data")
                mime_type = inline_data.get("mimeType") or inline_data.get("mime_type", "")
                if data and str(mime_type).startswith("image/"):
                    return base64.b64decode(data)

        raise RuntimeError("Image API did not return an image payload")
