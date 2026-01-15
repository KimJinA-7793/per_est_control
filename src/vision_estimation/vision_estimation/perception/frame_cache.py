# vision_estimation/perception/frame_cache.py
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class FramePacket:
    color: np.ndarray          # HxWx3 uint8 (BGR or RGB는 너 pipeline 기준)
    depth: np.ndarray          # HxW (dtype는 기존 depth.npy 기준)
    points_3d: np.ndarray      # Nx3 또는 HxWx3 (기존 click_points.Save_Cam() 형태에 맞춤)
    stamp_sec: float


class FrameCache:
    """
    Role:
      - Perception loop가 최신 프레임을 여기에 push
      - Estimation node는 여기서 최신 프레임을 pull (복사본)

    Thread safety:
      - writer(카메라 스레드) 1개, reader(ROS 콜백) 다수 가능
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest: Optional[FramePacket] = None

    def update(self, color: np.ndarray, depth: np.ndarray, points_3d: np.ndarray) -> None:
        pkt = FramePacket(
            color=color,
            depth=depth,
            points_3d=points_3d,
            stamp_sec=time.time(),
        )
        with self._lock:
            self._latest = pkt

    def get_latest_copy(self) -> Optional[FramePacket]:
        with self._lock:
            if self._latest is None:
                return None
            # ✅ reader가 pipeline에서 수정해도 안전하도록 복사
            return FramePacket(
                color=self._latest.color.copy(),
                depth=self._latest.depth.copy(),
                points_3d=np.array(self._latest.points_3d, copy=True),
                stamp_sec=float(self._latest.stamp_sec),
            )