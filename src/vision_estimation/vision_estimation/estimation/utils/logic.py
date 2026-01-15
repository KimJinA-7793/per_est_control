from .gemini_api import GeminiClient
from .utils import parse_labels_from_text, convert_labels_to_ids


class EstimationLogic:
    def __init__(self, prompt_cfg, api_key):
        self.cfg = prompt_cfg
        self.client = GeminiClient(
            api_key=api_key,
            model_name=self.cfg.default_model,
            timeout=self.cfg.default_timeout,
            temp=self.cfg.default_temp,
            max_tokens=self.cfg.default_max_tokens,
        )

    def run_inference(self, img_bytes: bytes, mime_type: str, expected_cnt: int, unknown_id: float) -> list[float]:
        """이미지를 받아 추론 후 ID 리스트 반환 (항상 길이=expected_cnt)."""
        try:
            prompt = self.cfg.get_prompt(expected_cnt)
            resp_text = self.client.generate(prompt, img_bytes, mime_type)
        except Exception as e:
            # 상위(Node)에서 logger로 바꾸는 것을 권장
            print(f"[Logic Error] Gemini API failed: {e}")
            return [unknown_id] * expected_cnt

        labels = parse_labels_from_text(resp_text)
        ids = convert_labels_to_ids(labels, self.cfg.label_to_id, unknown_id)

        # 길이 강제 (Add 단계 입력 무결성 보장)
        if len(ids) < expected_cnt:
            ids.extend([unknown_id] * (expected_cnt - len(ids)))
        return ids[:expected_cnt]