import json
import time
from dataclasses import dataclass

import requests
from requests.exceptions import ConnectionError, Timeout

from config import (
    get_script_api_base_url,
    get_script_api_key,
    get_script_api_model,
    get_verbose,
)
from status import warning


@dataclass
class ScriptApiProvider:
    timeout_seconds: int = 300
    max_retries: int = 5

    def generate_text(self, prompt: str, *, json_object: bool = False) -> str:
        base_url = get_script_api_base_url()
        api_key = get_script_api_key()
        model = get_script_api_model()

        if not base_url:
            raise RuntimeError("Missing config: script_api_base_url")
        if not api_key:
            raise RuntimeError("Missing config: script_api_key or SCRIPT_API_KEY")
        if not model:
            raise RuntimeError("Missing config: script_api_model")

        url = f"{base_url}/chat/completions"
        payload: dict = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.5 if json_object else 0.7,
        }
        if json_object:
            payload["response_format"] = {"type": "json_object"}
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        last_exc: Exception | None = None
        
        # 创建session并跳过代理
        session = requests.Session()
        session.trust_env = False  # 跳过环境变量中的代理设置
        
        for attempt in range(self.max_retries):
            try:
                response = session.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout_seconds,
                    proxies={"http": None, "https": None}  # 明确禁用代理
                )
                if json_object and response.status_code == 400:
                    try:
                        err_body = response.text[:400]
                    except Exception:
                        err_body = ""
                    if get_verbose():
                        warning(
                            "Script API rejected response_format json_object; retrying without it. "
                            f"HTTP {response.status_code} {err_body}"
                        )
                    payload.pop("response_format", None)
                    payload["temperature"] = 0.7
                    json_object = False
                    response = session.post(
                        url,
                        headers=headers,
                        json=payload,
                        timeout=self.timeout_seconds,
                        proxies={"http": None, "https": None}  # 明确禁用代理
                    )
                response.raise_for_status()
                data = response.json()
                try:
                    return str(data["choices"][0]["message"]["content"]).strip()
                except Exception as exc:
                    raise RuntimeError(
                        f"Script API returned unexpected payload: {json.dumps(data)[:500]}"
                    ) from exc
            except (ConnectionError, Timeout) as exc:
                last_exc = exc
                if attempt >= self.max_retries - 1:
                    break
                delay = min(30.0, 2.0**attempt)
                if get_verbose():
                    warning(
                        f"Script API connection failed ({exc!s}), retrying in {delay:.0f}s "
                        f"({attempt + 1}/{self.max_retries})..."
                    )
                time.sleep(delay)

        raise RuntimeError(
            "Script API unreachable after retries (connection reset, timeout, or TLS). "
            "Check network/VPN, firewall, and that script_api_base_url matches your provider "
            "(e.g. OpenAI-compatible base ending in /v1)."
        ) from last_exc
