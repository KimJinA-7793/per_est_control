from __future__ import annotations

import os
import numpy as np
from ament_index_python.packages import get_package_share_directory


class Coordinate:
    """
    Role:
      - pixel(u,v) + depth_frame -> world(x,y,z) 변환

    Data Flow:
      - camcalib.npz에서 camera_matrix, T_cam_to_work 로드
      - pixel_to_world(u,v,depth_frame) 계산

    Failure path:
      - calib 파일 없음 -> FileNotFoundError
      - 키 누락 -> KeyError
    """

    def __init__(self) -> None:
        self.camera_matrix: np.ndarray | None = None
        self.T_cam_to_work: np.ndarray | None = None
        self.load_calibration()

    @staticmethod
    def _get_calib_path() -> str:
        """
        ROS2 ament_python 정석:
          - 리소스는 site-packages(코드) 옆이 아니라 share 아래로 설치된다.
          - get_package_share_directory로 share 경로를 얻는다.
        """
        pkg_share = get_package_share_directory("vision_estimation")
        return os.path.join(
            pkg_share,
            "vision_estimation",
            "perception",
            "resources",
            "camcalib.npz",
        )

    def load_calibration(self) -> None:
        calib_path = self._get_calib_path()
        if not os.path.exists(calib_path):
            raise FileNotFoundError(f"camcalib.npz not found: {calib_path}")

        calib = np.load(calib_path)
        if "camera_matrix" not in calib or "T_cam_to_work" not in calib:
            raise KeyError(f"camcalib.npz missing keys: {calib.files}")

        self.camera_matrix = calib["camera_matrix"]
        self.T_cam_to_work = calib["T_cam_to_work"]

        self.fx = float(self.camera_matrix[0, 0])
        self.fy = float(self.camera_matrix[1, 1])
        self.cx = float(self.camera_matrix[0, 2])
        self.cy = float(self.camera_matrix[1, 2])

    def pixel_to_world(self, u: int, v: int, depth_frame) -> np.ndarray | None:
        """
        Input:
          - u,v: pixel
          - depth_frame: RealSense depth frame (has get_distance)

        Output:
          - np.ndarray shape(3,) [X,Y,Z] in work frame (meter)
          - depth invalid -> None

        Notes:
          - depth_frame.get_distance(u,v) returns meters (float)
        """
        if self.camera_matrix is None or self.T_cam_to_work is None:
            raise RuntimeError("Calibration not loaded")

        depth_m = float(depth_frame.get_distance(int(u), int(v)))
        if depth_m <= 0.0:
            return None

        Xc = (float(u) - self.cx) * depth_m / self.fx
        Yc = (float(v) - self.cy) * depth_m / self.fy
        Zc = depth_m

        Pc = np.array([Xc, Yc, Zc, 1.0], dtype=np.float64)
        Pw = self.T_cam_to_work @ Pc

        # 기존 프로젝트 보정: Z 부호 반전 유지 (필요하면 여기만 조정)
        Pw[2] = -Pw[2]

        return Pw[:3]