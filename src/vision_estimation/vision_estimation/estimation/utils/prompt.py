# prompt.py
# 목적:
# - unknown 최소화 (컨테이너면 무조건 plastic/can/paper/box 중 선택)
# - can / paper 구분을 "형태 특징"으로 강제
# - crop(객체 1개) 분류에 최적화

class PromptConfig:
    def __init__(self):
        self.default_model = "gemini-2.5-flash"
        self.default_timeout = 20.0
        self.default_temp = 0.0
        self.default_max_tokens = 1024

        self.allowed_labels = ["plastic", "can", "paper", "box", "unknown"]
        self.label_to_id = {
            "plastic": 0.0,
            "can": 1.0,
            "paper": 2.0,
            "box": 3.0,
            "unknown": -1.0,
        }

    # (전체 이미지용은 유지)
    def get_prompt(self, expected_count: int) -> str:
        return (
            "Return ONLY a JSON array of labels.\n"
            f"Length must be exactly {expected_count}.\n"
            "Allowed: [plastic, can, paper, box, unknown]\n"
            "No explanation.\n"
        )

    def get_prompt_single(self, is_blue_box: bool) -> str:
        """
        crop 1장(객체 1개) 분류용.
        - is_blue_box=True면 plastic 확정(투명컵 강제)
        """
        if is_blue_box:
            # 파란 박스(투명컵)는 논쟁 없이 plastic 고정
            return 'Return ONLY this exact JSON string: "plastic"'

        # 핵심: unknown 회피를 강제하고, can/paper 구분 특징을 매우 구체화
        return (
            "You see ONE object in an image crop for a robotic sorting system.\n"
            "Choose EXACTLY ONE label from this set:\n"
            "[\"plastic\",\"can\",\"paper\",\"box\",\"unknown\"].\n"
            "\n"
            "OUTPUT FORMAT:\n"
            "- Return ONLY a JSON string, like \"can\".\n"
            "- No extra text.\n"
            "\n"
            "CRITICAL DECISION RULE:\n"
            "- If the object is a CONTAINER (holds food/drink), you MUST NOT output \"unknown\".\n"
            "- \"unknown\" is allowed ONLY when the crop is empty background OR the object is not visible.\n"
            "\n"
            "CATEGORY RULES (use SHAPE cues more than color):\n"
            "\n"
            "1) can:\n"
            "- Cylindrical metal container.\n"
            "- Look for a ROUND TOP RIM and/or a PULL-TAB opening.\n"
            "- Even if the label is printed, if the body is cylindrical with metal top -> \"can\".\n"
            "\n"
            "2) paper:\n"
            "- Paper carton / pack (milk pack, yogurt pack, paper-based drink/food pack).\n"
            "- Typically a RECTANGULAR PRISM shape with folded edges/corners.\n"
            "- Often has printed paper surface; not shiny metal; not transparent.\n"
            "\n"
            "3) box:\n"
            "- Cardboard shipping box or brown paperboard box.\n"
            "- Large flat faces, brown fiber texture; not a drink container.\n"
            "\n"
            "4) plastic:\n"
            "- Transparent plastic cup/bottle/container.\n"
            "- Also plastic trays/containers.\n"
            "\n"
            "TIE-BREAKER (avoid unknown):\n"
            "- If it looks like a drink container and you see any round rim -> \"can\".\n"
            "- If it looks like a food/drink pack with straight edges -> \"paper\".\n"
        )