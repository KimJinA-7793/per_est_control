# gemini_api.py
# 통신 모듈 (Robust parsing + safer defaults)

import json
import urllib.request
import urllib.error
import base64
from typing import Any, Dict, Optional


class GeminiClient:
    def __init__(self, api_key, model_name, timeout, temp, max_tokens, force_json_mime: bool = False):
        self.api_key = api_key
        self.model_name = model_name
        self.timeout = float(timeout)
        self.temp = float(temp)
        self.max_tokens = int(max_tokens)
        self.force_json_mime = bool(force_json_mime)

    def generate(self, prompt: str, image_bytes: bytes, mime_type: str) -> str:
        """
        Input:
          - prompt(str), image_bytes(bytes), mime_type(str)
        Output:
          - model output text (string). (JSON-only는 prompt에서 강제)
        Failure path:
          - HTTPError/URLError -> RuntimeError
          - response parse fail -> RuntimeError(with finishReason/body)
        """

        if not self.api_key:
            raise ValueError("API key is missing (env: GEMINI_API_KEY)")

        encoded = base64.b64encode(image_bytes).decode("ascii")

        generation_cfg: Dict[str, Any] = {
            "temperature": self.temp,
            "maxOutputTokens": self.max_tokens,
        }
        # responseMimeType는 모델/엔드포인트 조합에 따라 content={}로 오는 케이스가 있어 기본 OFF
        if self.force_json_mime:
            generation_cfg["responseMimeType"] = "application/json"

        payload = {
            "contents": [{
                "role": "user",
                "parts": [
                    {"text": prompt},
                    {"inlineData": {"mimeType": mime_type, "data": encoded}},
                ],
            }],
            "generationConfig": generation_cfg,
        }

        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model_name}:generateContent?key={self.api_key}"
        )
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = e.read().decode("utf-8")
            except Exception:
                detail = "<no body>"
            raise RuntimeError(f"Gemini HTTPError code={e.code}, body={detail}")
        except urllib.error.URLError as e:
            raise RuntimeError(f"Gemini URLError reason={e.reason}")

        # ---- parse JSON ----
        try:
            data = json.loads(body)
        except Exception:
            raise RuntimeError(f"Gemini returned non-JSON body: {body}")

        text = self._extract_text(data)

        if text is None or text.strip() == "":
            finish = self._extract_finish_reason(data)
            # content가 {}로 오는 케이스 포함: 디버깅 가능한 정보까지 같이 던짐
            raise RuntimeError(
                "Invalid Gemini response: empty text. "
                f"finishReason={finish}, body={body}"
            )

        return text.strip()

    # -----------------------
    # Robust extract helpers
    # -----------------------
    def _extract_finish_reason(self, data: Dict[str, Any]) -> str:
        try:
            return str(data["candidates"][0].get("finishReason", ""))
        except Exception:
            return ""

    def _extract_text(self, data: Dict[str, Any]) -> Optional[str]:
        """
        Try multiple known response layouts.
        Returns:
          - concatenated text if found, else None
        """
        # 1) candidates[0].content.parts[].text (가장 일반)
        try:
            parts = data["candidates"][0]["content"]["parts"]
            texts = [p.get("text", "") for p in parts if isinstance(p, dict) and "text" in p]
            joined = "\n".join([t for t in texts if t])
            if joined.strip():
                return joined
        except Exception:
            pass

        # 2) candidates[0].content.text (혹시 단일 text로 오는 케이스)
        try:
            t = data["candidates"][0]["content"].get("text", "")
            if isinstance(t, str) and t.strip():
                return t
        except Exception:
            pass

        # 3) candidates[0].output (일부 래퍼/중계에서 쓰는 케이스)
        try:
            t = data["candidates"][0].get("output", "")
            if isinstance(t, str) and t.strip():
                return t
        except Exception:
            pass

        # 4) candidates[0].content is string (아주 예외)
        try:
            c = data["candidates"][0].get("content", None)
            if isinstance(c, str) and c.strip():
                return c
        except Exception:
            pass

        return None