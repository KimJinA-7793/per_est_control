# depth_utils.py
import numpy as np
from .coordinate import Coordinate


class FakeDepthFrameFromNpy:
    """
    depth.npy(z16, mm)를 RealSense depth_frame과 동일한 인터페이스로 사용하기 위한 래퍼
    """
    def __init__(self, depth_z16: np.ndarray):
        self.depth = depth_z16
        self.h, self.w = depth_z16.shape[:2]

    def get_distance(self, x: int, y: int) -> float:
        if x < 0 or y < 0 or x >= self.w or y >= self.h:
            return 0.0
        d_mm = float(self.depth[int(y), int(x)])  # z16 (mm)
        if d_mm <= 0:
            return 0.0
        return d_mm / 1000.0  # mm -> m


def box_center_pixel(box: np.ndarray) -> tuple[int, int]:
    """
    4점 폴리곤(회전 박스)의 중심 픽셀 계산
    """
    c = np.mean(box, axis=0)
    return int(round(c[0])), int(round(c[1]))


def boxes_to_world_list(green_items, depth_z16):
    """
    초록 박스 목록을 world 좌표 목록으로 변환

    green_items: [{"box": np.ndarray(4,2), "angle": float}, ...]
    depth_z16: HxW uint16 (mm)

    return:
      [{"id": i, "world": (X, Y, Z), "angle": angle_deg}, ...]
    """
    if not green_items or depth_z16 is None:
        return []

    fake_depth = FakeDepthFrameFromNpy(depth_z16)
    coord = Coordinate()

    out = []
    for i, item in enumerate(green_items):
        box = item["box"]
        angle = float(item["angle"])

        cx, cy = box_center_pixel(box)
        Pw = coord.pixel_to_world(cx, cy, fake_depth)
        if Pw is None:
            continue

        X, Y, Z = Pw[:3]
        out.append({
            "id": i,
            "world": (float(X), float(Y), float(Z)),
            "angle": angle,
        })

    return out


def _patch_valid_count(depth_z16: np.ndarray, u: int, v: int, k: int = 2) -> int:
    h, w = depth_z16.shape[:2]
    u0, u1 = max(0, u - k), min(w, u + k + 1)
    v0, v1 = max(0, v - k), min(h, v + k + 1)
    patch = depth_z16[v0:v1, u0:u1]
    return int(np.count_nonzero(patch > 0))


def _find_nearest_valid_pixel_in_rect(
    depth_z16: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    step: int = 2,
    max_r: int | None = None,
) -> tuple[int, int] | None:
    if depth_z16 is None or depth_z16.ndim != 2:
        return None

    h, w = depth_z16.shape[:2]
    x1 = int(max(0, min(w - 1, x1)))
    x2 = int(max(0, min(w - 1, x2)))
    y1 = int(max(0, min(h - 1, y1)))
    y2 = int(max(0, min(h - 1, y2)))
    if x2 <= x1 or y2 <= y1:
        return None

    cx = int(round((x1 + x2) / 2))
    cy = int(round((y1 + y2) / 2))

    if max_r is None:
        max_r = int(np.hypot((x2 - x1) / 2, (y2 - y1) / 2)) + 2

    if depth_z16[cy, cx] > 0:
        return (cx, cy)

    step = max(1, int(step))

    for r in range(step, max_r + 1, step):
        candidates: list[tuple[int, int]] = []

        top = cy - r
        bot = cy + r
        left = cx - r
        right = cx + r

        # 상/하 변
        for u in range(left, right + 1, step):
            if not (x1 <= u <= x2 and 0 <= u < w):
                continue
            if y1 <= top <= y2 and 0 <= top < h and depth_z16[top, u] > 0:
                candidates.append((u, top))
            if y1 <= bot <= y2 and 0 <= bot < h and depth_z16[bot, u] > 0:
                candidates.append((u, bot))

        # 좌/우 변 (중복 방지)
        for v in range(top + step, bot - step + 1, step):
            if not (y1 <= v <= y2 and 0 <= v < h):
                continue
            if x1 <= left <= x2 and 0 <= left < w and depth_z16[v, left] > 0:
                candidates.append((left, v))
            if x1 <= right <= x2 and 0 <= right < w and depth_z16[v, right] > 0:
                candidates.append((right, v))

        if candidates:
            best = None
            best_score = -1
            for (u, v) in candidates:
                score = _patch_valid_count(depth_z16, u, v, k=2)
                if score > best_score:
                    best_score = score
                    best = (u, v)
            return best

    return None


def blue_rect_to_world_safe(
    rect: tuple[int, int, int, int],
    depth_z16: np.ndarray,
    search_step: int = 2,
) -> np.ndarray | None:
    if depth_z16 is None:
        return None

    x1, y1, x2, y2 = rect

    fake_depth = FakeDepthFrameFromNpy(depth_z16)
    coord = Coordinate()

    cx = int(round((x1 + x2) / 2))
    cy = int(round((y1 + y2) / 2))

    # 1) 중심
    Pw = coord.pixel_to_world(cx, cy, fake_depth)
    if Pw is not None:
        return Pw

    # 2) 중심 근접 우선 탐색
    uv = _find_nearest_valid_pixel_in_rect(depth_z16, x1, y1, x2, y2, step=search_step)
    if uv is None:
        return None

    u, v = uv
    return coord.pixel_to_world(u, v, fake_depth)