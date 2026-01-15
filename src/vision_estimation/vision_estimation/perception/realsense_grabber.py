from __future__ import annotations

import numpy as np
import pyrealsense2 as rs


class RealSenseGrabber:
    """
    Role:
      - RealSense에서 color/depth/points_3d 한 세트를 제공한다.

    Output:
      - color: (H, W, 3) uint8 (BGR)  # OpenCV 기준
      - depth: (H, W) uint16         # RealSense raw depth
      - points_3d: (H, W, 3) float32 # meter 단위 (x,y,z)

    Failure path:
      - 프레임 수신 실패 시 RuntimeError
    """

    def __init__(self, width: int = 1280, height: int = 720, fps: int = 30) -> None:
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)

        self._initialized = False
        self._pipeline: rs.pipeline | None = None
        self._align: rs.align | None = None
        self._pc: rs.pointcloud | None = None

    def init(self) -> None:
        if self._initialized:
            return

        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)

        profile = pipeline.start(config)

        # depth -> color 좌표계로 정렬
        align = rs.align(rs.stream.color)

        # pointcloud 계산기
        pc = rs.pointcloud()

        # (옵션) auto-exposure 안정화: 5 프레임 드롭
        for _ in range(5):
            pipeline.wait_for_frames()

        self._pipeline = pipeline
        self._align = align
        self._pc = pc
        self._initialized = True

    def grab_once(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self._initialized or self._pipeline is None or self._align is None or self._pc is None:
            raise RuntimeError("RealSenseGrabber not initialized. Call init() first.")

        frames = self._pipeline.wait_for_frames()
        frames = self._align.process(frames)

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            raise RuntimeError("RealSense frame not ready (missing depth/color).")

        color = np.asanyarray(color_frame.get_data())  # (H,W,3) uint8 BGR
        depth = np.asanyarray(depth_frame.get_data())  # (H,W) uint16

        # point cloud (H*W vertices) -> (H,W,3)
        points = self._pc.calculate(depth_frame)

        # RealSense vertices는 structured array (f0,f1,f2)
        vtx = np.asanyarray(points.get_vertices())

        # view로 float32 3채널로 펼치기
        vtx = vtx.view(np.float32).reshape(-1, 3)

        points_3d = vtx.reshape(self.height, self.width, 3)

        h, w = depth.shape
        points_3d = vtx.reshape(h, w, 3)
        
        return color, depth, points_3d

    def close(self) -> None:
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception:
                pass
        self._pipeline = None
        self._align = None
        self._pc = None
        self._initialized = False