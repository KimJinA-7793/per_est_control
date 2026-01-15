#!/usr/bin/env python3
from __future__ import annotations

# ===== .env load (키 노출 방지) =====
from dotenv import load_dotenv, find_dotenv
import os

_dotenv_path = find_dotenv()
if not _dotenv_path:
    _dotenv_path = os.path.expanduser("~/0114/.env")
load_dotenv(dotenv_path=_dotenv_path, override=False)

import time
import threading
import traceback
from typing import List, Optional, Tuple

import numpy as np

import rclpy
from rclpy.node import Node

from rost_interfaces.srv import PerEstToControl
from rost_interfaces.msg import Float64Array

from vision_estimation.perception.frame_cache import FrameCache


class VisionEstimationNode(Node):
    """
    Spacebar snapshot -> crop classification(worker) -> control pulls via service.
    - Status: EMPTY -> PENDING -> READY/FAILED
    - queue=1: only latest snapshot kept
    - seq: increases per snapshot (spacebar)
    """

    def __init__(self) -> None:
        super().__init__("vision_estimation")

        # ---- Import check (pipeline) ----
        try:
            from vision_estimation.perception.utils import pipeline  # noqa: F401
            self.get_logger().info("pipeline import OK")
        except Exception:
            self.get_logger().error("pipeline import FAILED")
            self.get_logger().error(traceback.format_exc())
            raise

        self.get_logger().info(
            f"GEMINI_API_KEY loaded={bool(os.environ.get('GEMINI_API_KEY'))} | dotenv={_dotenv_path}"
        )

        self._cache = FrameCache()
        self._lock = threading.Lock()

        # ---- StepA: seq gate ----
        self._seq: int = 0
        self._latest_seq_ready: int = -1

        # ---- status ----
        self._status: str = "EMPTY"
        self._status_msg: str = "no snapshot yet"

        # ---- latest snapshot ----
        self._latest_internal: Optional[List[float]] = None
        self._latest_internal_stamp: Optional[float] = None
        self._latest_bins: Optional[List[List[float]]] = None
        self._latest_bins_stamp: Optional[float] = None
        self._latest_for_control: Optional[List[float]] = None
        self._latest_for_control_stamp: Optional[float] = None

        # ---- queue=1 pending ----
        self._req_id: int = 0
        self._pending_req_id: int = 0
        self._pending_seq: int = 0
        self._pending_cnt: int = 0
        self._pending_internal: Optional[List[float]] = None
        self._pending_crop_jpegs: Optional[List[bytes]] = None
        self._pending_crop_meta: Optional[List[Tuple[int, str]]] = None
        self._pending_bins: Optional[List[List[float]]] = None

        # ---- threads ----
        self._worker_thread = threading.Thread(target=self._classification_worker, daemon=True)
        self._worker_thread.start()
        self._cam_thread = threading.Thread(target=self._camera_loop, daemon=True)
        self._cam_thread.start()

        # ---- Health check timer (log on state change only) ----
        self._tick_count = 0
        self._last_logged_status: Optional[str] = None
        self._last_logged_seq: Optional[int] = None
        self._timer = self.create_timer(1.0, self._tick)

        # ---- Service ----
        self._srv = self.create_service(PerEstToControl, "per_est_to_control", self._on_per_est_to_control)
        self.get_logger().info("Service ready: /per_est_to_control (rost_interfaces/srv/PerEstToControl)")
        self.get_logger().info("VisionEstimationNode initialized")

    # =========================================================
    # Camera / GUI loop
    # =========================================================
    def _camera_loop(self) -> None:
        from vision_estimation.perception.utils import realsense_loop
        from vision_estimation.perception.utils import click_points
        from vision_estimation.perception.utils import pipeline

        def update_color_image(color_img):
            click_points.update_color_image(color_img)

        def update_depth_frame(depth_frame):
            click_points.update_depth_frame(depth_frame)
            try:
                depth_np = np.asanyarray(depth_frame.get_data())
                color_np = click_points.color_image_global
                if color_np is not None:
                    self._cache.update(
                        color=color_np,
                        depth=depth_np,
                        points_3d=np.empty((0, 3), dtype=np.float32),
                    )
            except Exception as e:
                self.get_logger().error(f"FrameCache update failed: {e}")

        def on_spacebar_save():
            """
            Spacebar snapshot:
              - pipeline.save_cam() must return:
                  color (raw), flat_world_list, bboxes_xyxy_sorted, flat_clicked_xy
              - assign seq and push queue=1 to worker
            """
            result = pipeline.save_cam()
            if not isinstance(result, dict):
                self.get_logger().warn("save_cam returned invalid result")
                return

            color = result.get("color", None)
            flat_world = result.get("flat_world_list", None)
            bboxes = result.get("bboxes_xyxy_sorted", None)
            bins_xy = result.get("flat_clicked_xy", None)

            if color is None or flat_world is None or bboxes is None:
                self.get_logger().warn("missing color/flat_world_list/bboxes_xyxy_sorted")
                return

            if bins_xy is None or not isinstance(bins_xy, list):
                bins_xy = []

            if len(flat_world) % 5 != 0:
                self.get_logger().warn(f"flat_world_list length not multiple of 5: {len(flat_world)}")

            expected_cnt = len(flat_world) // 5
            if len(bboxes) != expected_cnt:
                with self._lock:
                    self._status = "FAILED"
                    self._status_msg = "bbox count mismatch"
                self.get_logger().error(f"bbox count mismatch: {len(bboxes)} vs {expected_cnt}")
                return

            # internal: [tmp_id, x, y, z, angle] * N
            internal: List[float] = []
            for i in range(0, len(flat_world), 5):
                tmp_id = int(flat_world[i + 0])
                x = float(flat_world[i + 1])
                y = float(flat_world[i + 2])
                z = float(flat_world[i + 3])
                ang = float(flat_world[i + 4])
                internal.extend([tmp_id, x, y, z, ang])

            # crop jpegs
            import cv2

            H, W = color.shape[:2]
            crop_jpegs: List[bytes] = []
            crop_meta: List[Tuple[int, str]] = []

            for (obj_id, x1, y1, x2, y2, obj_type) in bboxes:
                x1 = max(0, min(W - 1, int(x1)))
                y1 = max(0, min(H - 1, int(y1)))
                x2 = max(0, min(W, int(x2)))
                y2 = max(0, min(H, int(y2)))

                if x2 <= x1 or y2 <= y1:
                    self.get_logger().warn(f"invalid bbox for id={obj_id}: {(x1,y1,x2,y2)}")
                    crop_jpegs.append(b"")
                    crop_meta.append((int(obj_id), str(obj_type)))
                    continue

                crop = color[y1:y2, x1:x2].copy()
                ok, buf = cv2.imencode(".jpg", crop)
                crop_jpegs.append(buf.tobytes() if ok else b"")
                crop_meta.append((int(obj_id), str(obj_type)))

            now = time.time()
            with self._lock:
                self._seq += 1
                snap_seq = self._seq

                self._latest_internal = internal
                self._latest_internal_stamp = now
                self._latest_bins = [[float(p[0]), float(p[1])] for p in bins_xy] if bins_xy else []
                self._latest_bins_stamp = now

                self._latest_for_control = None
                self._latest_for_control_stamp = None

                self._status = "PENDING"
                self._status_msg = f"classification pending (objs={expected_cnt})"

                # queue=1
                self._req_id += 1
                self._pending_req_id = self._req_id
                self._pending_seq = snap_seq
                self._pending_cnt = expected_cnt
                self._pending_internal = internal
                self._pending_crop_jpegs = crop_jpegs
                self._pending_crop_meta = crop_meta
                self._pending_bins = self._latest_bins

            self.get_logger().info(f"Snapshot stored: seq={snap_seq}, objs={expected_cnt}, bins={len(bins_xy)}")

        realsense_loop.run(
            width=1280,
            height=720,
            fps=30,
            on_save=on_spacebar_save,
            on_reset=click_points.reset_points,
            on_click=click_points.mouse_callback,
            update_depth_frame=update_depth_frame,
            update_color_image=update_color_image,
            get_points=click_points.get_saved_points,
        )

    # =========================================================
    # Worker: Gemini classify per crop (queue=1)
    # =========================================================
    def _classification_worker(self) -> None:
        # PromptConfig
        try:
            from vision_estimation.estimation.utils.prompt import PromptConfig
        except Exception:
            PromptConfig = None  # type: ignore

        # GeminiClient (two possible locations)
        GeminiClient = None
        try:
            from vision_estimation.adapters.gemini_api import GeminiClient  # type: ignore
        except Exception:
            try:
                from vision_estimation.estimation.utils.gemini_api import GeminiClient  # type: ignore
            except Exception:
                GeminiClient = None  # type: ignore

        api_key = (
            os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
            or os.environ.get("GEMINI_APIKEY")
        )

        import json

        def parse_label(text: str) -> str:
            """
            Expect: JSON array length 1, e.g. ["paper"].
            Fallback: plain string.
            """
            s = (text or "").strip()
            try:
                v = json.loads(s)
                if isinstance(v, list) and len(v) >= 1 and isinstance(v[0], str):
                    return v[0].strip().lower()
                if isinstance(v, str):
                    return v.strip().lower()
            except Exception:
                pass
            return s.strip().strip('"').strip().lower()

        while rclpy.ok():
            with self._lock:
                req_id = self._pending_req_id
                my_seq = self._pending_seq
                cnt = self._pending_cnt
                internal = self._pending_internal
                crop_jpegs = self._pending_crop_jpegs
                crop_meta = self._pending_crop_meta
                bins = self._pending_bins

            if req_id == 0 or internal is None or crop_jpegs is None or crop_meta is None:
                time.sleep(0.05)
                continue

            my_req = req_id
            unknown_id = -1.0
            type_ids: List[float] = [unknown_id] * cnt

            try:
                if PromptConfig is None:
                    raise RuntimeError("PromptConfig import failed")
                if GeminiClient is None:
                    raise RuntimeError("GeminiClient import failed")
                if not api_key:
                    raise RuntimeError("GEMINI_API_KEY not set in environment")

                cfg = PromptConfig()
                client = GeminiClient(
                    api_key=api_key,
                    model_name=cfg.default_model,
                    timeout=cfg.default_timeout,
                    temp=cfg.default_temp,
                    max_tokens=cfg.default_max_tokens,
                )

                # per-crop classification
                for i in range(cnt):
                    _, obj_type = crop_meta[i]
                    is_blue = (str(obj_type).lower() == "blue")

                    # blue box rule
                    if is_blue:
                        type_ids[i] = float(cfg.label_to_id["plastic"])
                        continue

                    img = crop_jpegs[i]
                    if not img:
                        type_ids[i] = unknown_id
                        continue

                    prompt = cfg.get_prompt(expected_count=1)
                    resp_text = client.generate(prompt, img, "image/jpeg")
                    label = parse_label(resp_text)

                    if label not in cfg.allowed_labels:
                        label = "unknown"
                    type_ids[i] = float(cfg.label_to_id.get(label, unknown_id))

            except Exception as e:
                self.get_logger().error(f"classification failed: {e}")

            with self._lock:
                # drop if newer snapshot arrived
                if self._pending_req_id != my_req:
                    continue

                # build control flat: [type_id, x, y, z, angle] * N
                control_flat: List[float] = []
                for k in range(cnt):
                    base = k * 5
                    x = float(internal[base + 1])
                    y = float(internal[base + 2])
                    z = float(internal[base + 3])
                    ang = float(internal[base + 4])
                    control_flat.extend([float(type_ids[k]), x, y, z, ang])

                self._latest_for_control = control_flat
                self._latest_for_control_stamp = time.time()

                if bins is not None:
                    self._latest_bins = bins
                    self._latest_bins_stamp = time.time()

                if all(t == -1.0 for t in type_ids):
                    self._status = "FAILED"
                    self._status_msg = "all unknown (check API / prompt / crop)"
                else:
                    self._status = "READY"
                    self._status_msg = f"ready (objs={cnt})"
                    self._latest_seq_ready = int(my_seq)

                # consume pending
                self._pending_req_id = 0
                self._pending_seq = 0
                self._pending_cnt = 0
                self._pending_internal = None
                self._pending_crop_jpegs = None
                self._pending_crop_meta = None
                self._pending_bins = None

            time.sleep(0.01)

    # =========================================================
    # Health check (log only on change)
    # =========================================================
    def _tick(self) -> None:
        self._tick_count += 1
        if self._tick_count % 5 != 0:
            return

        pkt = self._cache.get_latest_copy()
        if pkt is None:
            if self._last_logged_status != "NO_FRAME":
                self._last_logged_status = "NO_FRAME"
                self._last_logged_seq = None
                self.get_logger().warn("no frame yet")
            return

        age_ms = (time.time() - pkt.stamp_sec) * 1000.0
        with self._lock:
            status = self._status
            status_msg = self._status_msg
            seq_now = int(self._seq)

        # ✅ 상태 또는 seq가 바뀐 경우에만 로그
        if status != self._last_logged_status or seq_now != self._last_logged_seq:
            self._last_logged_status = status
            self._last_logged_seq = seq_now
            self.get_logger().info(f"frame ok | age={age_ms:.1f} ms | status={status} ({status_msg})")

    # =========================================================
    # Service: /per_est_to_control
    # =========================================================
    def _on_per_est_to_control(self, request, response):
        with self._lock:
            status = self._status
            status_msg = self._status_msg
            seq_now = int(self._seq)
            seq_ready = int(self._latest_seq_ready)
            flat = self._latest_for_control
            flat_stamp = self._latest_for_control_stamp
            bins = self._latest_bins

        # not ready
        if status != "READY" or flat is None or flat_stamp is None:
            response.seq = seq_now
            response.success = False
            response.message = f"not ready: status={status}, msg={status_msg}"
            response.trash_list = []
            response.bin_list = []
            return response

        # ready
        response.seq = seq_ready

        trash_msg = Float64Array()
        trash_msg.data = [float(v) for v in flat]
        response.trash_list = [trash_msg]

        bin_msgs: List[Float64Array] = []
        if bins:
            for xy in bins:
                if not xy or len(xy) < 2:
                    continue
                m = Float64Array()
                m.data = [float(xy[0]), float(xy[1])]
                bin_msgs.append(m)
        response.bin_list = bin_msgs

        age_ms = (time.time() - flat_stamp) * 1000.0
        response.success = True
        response.message = f"READY | seq={seq_ready} | objs={len(flat)//5} | bins={len(bin_msgs)} | age_ms={age_ms:.1f}"
        return response


def main(argv: Optional[list[str]] = None) -> int:
    rclpy.init(args=argv)
    node: Optional[VisionEstimationNode] = None
    try:
        node = VisionEstimationNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception:
        print(traceback.format_exc())
        return 1
    finally:
        if node is not None:
            node.destroy_node()
        # shutdown 중복 방지
        if rclpy.ok():
            rclpy.shutdown()
    return 0