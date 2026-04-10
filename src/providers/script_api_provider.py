import json
from dataclasses import dataclass

import requests

from config import get_script_api_base_url, get_script_api_key, get_script_api_model


@dataclass
class ScriptApiProvider:
    timeout_seconds: int = 120

    def generate_text(self, prompt: str) -> str:
        base_url = get_script_api_base_url()
        api_key = get_script_api_key()
        model = get_script_api_model()

        if not base_url:
            raise RuntimeError("Missing config: script_api_base_url")
        if not api_key:
            raise RuntimeError("Missing config: script_api_key or SCRIPT_API_KEY")
        if not model:
            raise RuntimeError("Missing config: script_api_model")

        response = requests.post(
            f"{base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
            },
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        data = response.json()
        try:
            return str(data["choices"][0]["message"]["content"]).strip()
        except Exception as exc:
            raise RuntimeError(
                f"Script API returned unexpected payload: {json.dumps(data)[:500]}"
            ) from exc
