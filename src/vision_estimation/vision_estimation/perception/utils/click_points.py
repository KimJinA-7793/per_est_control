# click_points.py
import cv2
import numpy as np
from .coordinate import Coordinate

depth_frame_global = None
color_image_global = None
current_points = []


def update_depth_frame(frame):
    global depth_frame_global
    depth_frame_global = frame


def update_color_image(frame):
    global color_image_global
    color_image_global = frame


def mouse_callback(event, x, y, flags, param):
    global current_points, depth_frame_global
    if event == cv2.EVENT_LBUTTONDOWN:
        if depth_frame_global is None:
            print("Depth 프레임 없음")
            return
        depth_mm = depth_frame_global.get_distance(x, y) * 1000.0
        current_points.append((x, y, depth_mm))
        print(f"Click: x={x}, y={y}, depth={depth_mm:.1f} mm")


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
        d = float(self.depth[int(y), int(x)])  # z16 in mm
        if d <= 0:
            return 0.0
        return d / 1000.0  # mm -> m


def Save_Cam():
    """
    저장 버튼을 누른 시점의 프레임을 스냅샷으로 고정하고,
    누적된 클릭 포인트를 해당 depth 기준 world 좌표로 변환
    """
    global depth_frame_global, color_image_global, current_points

    if depth_frame_global is None or color_image_global is None:
        print("프레임 없음")
        return None, None, []

    color_image = color_image_global.copy()
    depth_image = np.asanyarray(depth_frame_global.get_data()).copy()

    fake_depth_frame = FakeDepthFrameFromNpy(depth_image)

    picker = Coordinate()
    points_3d = []

    for u, v, _ in current_points:
        Pw = picker.pixel_to_world(u, v, fake_depth_frame)
        if Pw is not None:
            points_3d.append(Pw[:3])

    if points_3d:
        print(f"3D 좌표 계산 완료 ({len(points_3d)}개 점)")
    else:
        print("저장할 3D 좌표 없음")

    return color_image, depth_image, points_3d


def reset_points():
    global current_points
    current_points.clear()
    print("포인트 리셋 완료")


def get_saved_points():
    return current_points