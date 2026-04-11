#!/usr/bin/env python3
"""
Call the same OpenAI-compatible image API as MoneyPrinterV2 using the real `curl` binary.

Reads nanobanana2_* from repo-root config.json (or GEMINI_API_KEY if key empty).

Usage (from repo root):
  python3 scripts/curl_one_image.py
  python3 scripts/curl_one_image.py "your prompt here"
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "prompt",
        nargs="?",
        default="A red lantern in snow at night, cinematic wide shot, soft light.",
    )
    args = parser.parse_args()

    cfg_path = os.path.join(_REPO, "config.json")
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)

    base = str(cfg.get("nanobanana2_api_base_url", "")).rstrip("/")
    key = str(cfg.get("nanobanana2_api_key", "")).strip() or os.environ.get("GEMINI_API_KEY", "").strip()
    model = str(cfg.get("nanobanana2_model", "gemini-3.1-flash-image-preview")).strip()
    size = str(cfg.get("nanobanana2_aspect_ratio", "16:9")).strip() or "16:9"
    timeout = int(cfg.get("nanobanana2_image_timeout_seconds", 300))

    if not base:
        print("config.json: nanobanana2_api_base_url is empty", file=sys.stderr)
        return 1
    if not key:
        print("config.json nanobanana2_api_key and GEMINI_API_KEY are both empty", file=sys.stderr)
        return 1

    url = f"{base}/v1/images/generations"
    body = {
        "model": model,
        "prompt": args.prompt,
        "n": 1,
        "size": size,
        "response_format": "b64_json",
    }
    payload = json.dumps(body, ensure_ascii=False)

    mp_dir = os.path.join(_REPO, ".mp")
    os.makedirs(mp_dir, exist_ok=True)
    resp_path = os.path.join(mp_dir, "curl_image_response.json")

    cmd = [
        "curl",
        "-sS",
        "-m",
        str(timeout),
        "-X",
        "POST",
        url,
        "-H",
        f"Authorization: Bearer {key}",
        "-H",
        "Content-Type: application/json; charset=utf-8",
        "-d",
        payload,
        "-o",
        resp_path,
        "-w",
        "\nHTTP_CODE:%{http_code}\n",
    ]
    print("[curl_one_image] POST", url)
    print("[curl_one_image] model=", model, "size=", size, "timeout=", timeout, "s", sep="")

    proc = subprocess.run(cmd, cwd=_REPO)
    if proc.returncode != 0:
        print("[curl_one_image] curl failed with exit", proc.returncode, file=sys.stderr)
        return proc.returncode

    with open(resp_path, encoding="utf-8") as f:
        text = f.read()
    if "HTTP_CODE:" in text:
        body_text, _, code_part = text.rpartition("HTTP_CODE:")
        code = code_part.strip()
        if code != "200":
            print("[curl_one_image] non-200 response saved to", resp_path, file=sys.stderr)
            print(body_text[:2000], file=sys.stderr)
            return 1
        text = body_text

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        print("[curl_one_image] invalid JSON in response file:", exc, file=sys.stderr)
        print(text[:1500], file=sys.stderr)
        return 1

    try:
        raw_b64 = data["data"][0]["b64_json"]
    except (KeyError, IndexError) as exc:
        print("[curl_one_image] unexpected JSON shape:", exc, file=sys.stderr)
        print(json.dumps(data)[:1500], file=sys.stderr)
        return 1

    raw_b64 = str(raw_b64).strip()
    if raw_b64.startswith("data:"):
        raw_b64 = raw_b64.split(",", 1)[1]
    pad = len(raw_b64) % 4
    if pad:
        raw_b64 += "=" * (4 - pad)
    img = base64.b64decode(raw_b64)

    if img[:3] == b"\xff\xd8\xff":
        out = os.path.join(mp_dir, "curl_image_out.jpg")
    elif img[:4] == b"\x89PNG":
        out = os.path.join(mp_dir, "curl_image_out.png")
    else:
        out = os.path.join(mp_dir, "curl_image_out.bin")

    with open(out, "wb") as f:
        f.write(img)
    print("[curl_one_image] saved", out, f"({len(img)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
