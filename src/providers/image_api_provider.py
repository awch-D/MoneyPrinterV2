import base64

import requests

from config import (
    get_nanobanana2_api_base_url,
    get_nanobanana2_api_key,
    get_nanobanana2_aspect_ratio,
    get_nanobanana2_model,
)


class ImageApiProvider:
    def __init__(self, timeout_seconds: int = 300):
        self.timeout_seconds = timeout_seconds

    def generate_image_bytes(self, prompt: str) -> bytes:
        api_key = get_nanobanana2_api_key()
        if not api_key:
            raise RuntimeError("Missing config: nanobanana2_api_key")

        endpoint = f"{get_nanobanana2_api_base_url()}/v1/images/generations"
        payload = {
            "model": get_nanobanana2_model(),
            "prompt": prompt,
            "n": 1,
            "size": get_nanobanana2_aspect_ratio(),
            "response_format": "b64_json",
        }

        response = requests.post(
            endpoint,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        body = response.json()

        b64 = body["data"][0]["b64_json"]
        # 剥离 data URI 前缀（如 data:image/jpeg;base64,）
        if "," in b64:
            b64 = b64.split(",", 1)[1]
        return base64.b64decode(b64)
