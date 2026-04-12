import base64
import re
import time

import requests
from requests.exceptions import ConnectionError, HTTPError, Timeout

from config import (
    get_nanobanana2_api_base_url,
    get_nanobanana2_api_key,
    get_nanobanana2_aspect_ratio,
    get_nanobanana2_ignore_env_proxy,
    get_nanobanana2_image_max_retries,
    get_nanobanana2_image_timeout_seconds,
    get_nanobanana2_model,
)
from status import warning

# OpenAI-compatible Gemini image proxies expect ratio strings, not pixel sizes (400 otherwise).
# Aligned with Novel2Toon `gemini-image-gen_副本` / SKILL.md.
_ALLOWED_ASPECTS = frozenset(
    {
        "1:1",
        "1:4",
        "1:8",
        "2:3",
        "3:2",
        "3:4",
        "4:1",
        "4:3",
        "4:5",
        "5:4",
        "8:1",
        "9:16",
        "16:9",
        "21:9",
    }
)


def _normalize_image_size_parameter(raw: str) -> str:
    s = (raw or "").strip().replace("X", "x").replace("×", "x")
    low = s.lower()
    if low in ("portrait", "vertical"):
        return "9:16"
    if low in ("landscape", "horizontal", "wide"):
        return "16:9"
    if low in _ALLOWED_ASPECTS:
        return low
    m = re.match(r"^(\d+)\s*x\s*(\d+)$", low)
    if m:
        w, h = int(m.group(1)), int(m.group(2))
        r = w / max(h, 1)
        if abs(r - 16 / 9) < 0.02:
            warning(
                f'Image size "{raw}" looks like pixel dimensions; using "16:9" '
                "(Gemini proxy expects ratio strings like 16:9, not 1920x1080)."
            )
            return "16:9"
        if abs(r - 9 / 16) < 0.02:
            warning(f'Image size "{raw}" looks like pixel dimensions; using "9:16".')
            return "9:16"
        if abs(r - 1.0) < 0.02:
            return "1:1"
    warning(f'nanobanana2_aspect_ratio {raw!r} is not a supported ratio string; using "16:9".')
    return "16:9"


def _decode_b64_json_field(raw: str) -> bytes:
    """Strip data URI prefix and fix base64 padding (matches gemini-image-gen scripts/generate.py)."""
    raw_b64 = (raw or "").strip()
    if raw_b64.startswith("data:"):
        raw_b64 = raw_b64.split(",", 1)[1]
    elif "," in raw_b64:
        raw_b64 = raw_b64.split(",", 1)[1]
    pad = len(raw_b64) % 4
    if pad:
        raw_b64 += "=" * (4 - pad)
    return base64.b64decode(raw_b64)


def _image_bytes_from_generation_response(
    body: dict, *, url_timeout: int = 120, session: requests.Session | None = None
) -> bytes:
    item = body["data"][0]
    if "b64_json" in item and item["b64_json"]:
        return _decode_b64_json_field(str(item["b64_json"]))
    if "url" in item and item["url"]:
        sess = session or requests.Session()
        r = sess.get(str(item["url"]), timeout=url_timeout)
        r.raise_for_status()
        return r.content
    raise RuntimeError(f"Image API response missing b64_json and url: {list(item.keys())}")


class ImageApiProvider:
    def __init__(self, timeout_seconds: int | None = None):
        self.timeout_seconds = (
            timeout_seconds
            if timeout_seconds is not None
            else get_nanobanana2_image_timeout_seconds()
        )

    def generate_image_bytes(self, prompt: str, *, aspect_ratio: str | None = None) -> bytes:
        api_key = get_nanobanana2_api_key()
        if not api_key:
            raise RuntimeError("Missing config: nanobanana2_api_key")

        raw_size = (aspect_ratio or "").strip() or get_nanobanana2_aspect_ratio()
        size = _normalize_image_size_parameter(raw_size)
        endpoint = f"{get_nanobanana2_api_base_url()}/v1/images/generations"
        payload = {
            "model": get_nanobanana2_model(),
            "prompt": prompt,
            "n": 1,
            "size": size,
            "response_format": "b64_json",
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        timeout = self.timeout_seconds
        max_retries = get_nanobanana2_image_max_retries()
        session = requests.Session()
        if get_nanobanana2_ignore_env_proxy():
            session.trust_env = False
        last_exc: Exception | None = None

        for attempt in range(max_retries):
            try:
                response = session.post(
                    endpoint,
                    headers=headers,
                    json=payload,
                    timeout=timeout,
                )
                response.raise_for_status()
                body = response.json()
                return _image_bytes_from_generation_response(body, session=session)
            except (Timeout, ConnectionError) as exc:
                last_exc = exc
                warning(
                    f"Image API request failed ({type(exc).__name__}: {exc}); "
                    f"retry {attempt + 1}/{max_retries} after {timeout}s timeout window..."
                )
            except HTTPError as exc:
                last_exc = exc
                code = exc.response.status_code if exc.response is not None else 0
                if code in (429, 502, 503, 504) and attempt < max_retries - 1:
                    warning(f"Image API HTTP {code}; retry {attempt + 1}/{max_retries}...")
                else:
                    raise
            if attempt < max_retries - 1:
                time.sleep(min(60.0, 2.0**attempt))

        raise RuntimeError(f"Image API failed after {max_retries} attempts: {last_exc!s}") from last_exc
