#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional
import os
import time
import threading
import queue

import rclpy
from rclpy.node import Node

from rost_interfaces.srv import PerEstToControl
from dsr_msgs2.srv import MoveJoint


@dataclass
class TrashItem:
    type_id: int
    x: float
    y: float
    z: float
    angle: float


class ControlRunner(Node):
    """
    Role (system integration):
      - Pull estimation results via service (/per_est_to_control) at 2Hz (non-blocking)
      - When READY and seq changes: enqueue a robot motion job (queue=1)
      - Robot motion is executed in a worker thread via ROS2 service:
          /dsr01/motion/move_joint (dsr_msgs2/srv/MoveJoint)

    Data Flow:
      PerEstToControl.Response.trash_list[0].data (flat: [type,x,y,z,ang]*N)
        -> parse -> List[TrashItem]
        -> (policy) choose a joint pose
        -> MoveJoint.Request(pos[6], vel, acc, time, radius, mode, blend_type, sync_type)
        -> call_async -> MoveJoint.Response(success)

    Control Flow:
      timer(2Hz): poll service (no blocking)
      READY(new seq): enqueue job
      worker thread: wait for move_joint service, send request, wait up to motion_timeout_sec
    """

    def __init__(self) -> None:
        super().__init__("control_runner")

        # ---- Perception/Estimation pull ----
        self.service_name = "/per_est_to_control"
        self.poll_hz = 2.0
        self.poll_timeout_sec = 2.0
        self.cli = self.create_client(PerEstToControl, self.service_name)

        # ---- Doosan motion service (direct) ----
        self.move_joint_srv = "/dsr01/motion/move_joint"
        self.movej_cli = self.create_client(MoveJoint, self.move_joint_srv)

        # ---- Motion policy params (safe defaults) ----
        # IMPORTANT: This code moves the robot. Gate with env var.
        # export ENABLE_MOTION=1 to actually move.
        self.enable_motion = os.environ.get("ENABLE_MOTION", "0") == "1"

        # Joint target for a simple "smoke test" move.
        # You MUST adjust to your cell/robot limits. Unit: usually degrees in Doosan ROS2 examples.
        self.test_posj: List[float] = self._parse_6f_env("TEST_POSJ", default=[0, 0, 90, 0, 90, 0])

        # Motion numeric settings
        self.vel = float(os.environ.get("MOVEJ_VEL", "20.0"))
        self.acc = float(os.environ.get("MOVEJ_ACC", "20.0"))
        self.time_s = float(os.environ.get("MOVEJ_TIME", "0.0"))      # 0 => use vel/acc
        self.radius = float(os.environ.get("MOVEJ_RADIUS", "0.0"))    # blending radius
        self.mode = int(os.environ.get("MOVEJ_MODE", "0"))
        self.blend_type = int(os.environ.get("MOVEJ_BLEND", "0"))
        self.sync_type = int(os.environ.get("MOVEJ_SYNC", "0"))
        self.motion_timeout_sec = float(os.environ.get("MOTION_TIMEOUT", "10.0"))

        # ---- seq gating ----
        self._pending_future = None
        self._pending_sent_at = 0.0
        self._last_seq_logged: int = -1
        self._last_seq_executed: int = -1

        # spam reduction
        self._last_not_ready_msg: str = ""
        self._last_warn_state: str = ""  # "not_ready_service" / "timeout" / ""

        # ---- robot worker (queue=1) ----
        self._job_q: "queue.Queue[Tuple[int, List[TrashItem], List[Tuple[float, float]]]]" = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._robot_thread = threading.Thread(target=self._robot_worker, daemon=True)
        self._robot_thread.start()

        self.get_logger().info(
            f"ControlRunner started | poll={self.poll_hz:.0f}Hz | motion_service={self.move_joint_srv} "
            f"| ENABLE_MOTION={int(self.enable_motion)}"
        )
        self.timer = self.create_timer(1.0 / self.poll_hz, self._on_timer)

    # =========================================================
    # Helpers
    # =========================================================
    def _parse_6f_env(self, key: str, default: List[float]) -> List[float]:
        raw = os.environ.get(key, "")
        if not raw.strip():
            return list(default)
        try:
            parts = [p.strip() for p in raw.replace("[", "").replace("]", "").split(",")]
            vals = [float(p) for p in parts if p != ""]
            if len(vals) != 6:
                self.get_logger().warn(f"{key} must have 6 numbers, got {len(vals)}. Using default.")
                return list(default)
            return vals
        except Exception:
            self.get_logger().warn(f"Failed to parse {key}. Using default.")
            return list(default)

    def _warn_once(self, state: str, msg: str) -> None:
        if self._last_warn_state != state:
            self._last_warn_state = state
            self.get_logger().warn(msg)

    def _clear_warn_state(self) -> None:
        self._last_warn_state = ""

    # =========================================================
    # Poll loop (non-blocking)
    # =========================================================
    def _on_timer(self) -> None:
        # 0) Estimation service ready?
        if not self.cli.service_is_ready():
            self.cli.wait_for_service(timeout_sec=0.1)
            self._warn_once("not_ready_service", "per_est_to_control service not available yet")
            return

        # 1) handle pending future
        if self._pending_future is not None:
            if self._pending_future.done():
                try:
                    resp = self._pending_future.result()
                except Exception as e:
                    self.get_logger().error(f"per_est_to_control call failed: {e}")
                    self._pending_future = None
                    self._pending_sent_at = 0.0
                    return

                self._pending_future = None
                self._pending_sent_at = 0.0
                self._handle_response(resp)
                return

            # timeout
            if (time.time() - self._pending_sent_at) > self.poll_timeout_sec:
                self._warn_once("timeout", "per_est_to_control timeout/no response")
                self._pending_future = None
                self._pending_sent_at = 0.0
            return

        # 2) send new request
        req = PerEstToControl.Request()
        self._pending_future = self.cli.call_async(req)
        self._pending_sent_at = time.time()

    # =========================================================
    # Response handling
    # =========================================================
    def _handle_response(self, resp: PerEstToControl.Response) -> None:
        self._clear_warn_state()

        if not resp.success:
            if resp.message != self._last_not_ready_msg:
                self._last_not_ready_msg = resp.message
                self.get_logger().info(f"estimation not ready: {resp.message}")
            return

        self._last_not_ready_msg = ""
        seq = int(getattr(resp, "seq", -1))

        trash_items = self._parse_trash(resp)
        bins = self._parse_bins(resp)

        if seq != self._last_seq_logged:
            self._last_seq_logged = seq
            self.get_logger().info(f"recv {resp.message}")

        # only trigger on new seq
        if seq != self._last_seq_executed:
            self._last_seq_executed = seq
            self.execute(trash_items, bins)

    def _parse_trash(self, resp: PerEstToControl.Response) -> List[TrashItem]:
        if not resp.trash_list:
            return []
        flat = list(resp.trash_list[0].data)
        if len(flat) % 5 != 0:
            self.get_logger().warn(f"trash_list length not multiple of 5: {len(flat)}")

        items: List[TrashItem] = []
        for i in range(0, len(flat) - 4, 5):
            items.append(
                TrashItem(
                    type_id=int(flat[i + 0]),
                    x=float(flat[i + 1]),
                    y=float(flat[i + 2]),
                    z=float(flat[i + 3]),
                    angle=float(flat[i + 4]),
                )
            )
        return items

    def _parse_bins(self, resp: PerEstToControl.Response) -> List[Tuple[float, float]]:
        out: List[Tuple[float, float]] = []
        for arr in resp.bin_list:
            if len(arr.data) >= 2:
                out.append((float(arr.data[0]), float(arr.data[1])))
        return out

    # =========================================================
    # Execute (enqueue only; non-blocking)
    # =========================================================
    def execute(self, trash: List[TrashItem], bins: List[Tuple[float, float]]) -> None:
        self.get_logger().info(
            f"EXEC(plan) | objs={len(trash)} | trash={[[t.type_id,t.x,t.y,t.z,t.angle] for t in trash]} | bins={bins}"
        )

        # queue=1: keep only latest job
        job = (self._last_seq_executed, trash, bins)
        try:
            self._job_q.put_nowait(job)
        except queue.Full:
            try:
                _ = self._job_q.get_nowait()
            except Exception:
                pass
            try:
                self._job_q.put_nowait(job)
            except queue.Full:
                self.get_logger().warn("robot job queue still full; drop")

    # =========================================================
    # Robot worker: call /dsr01/motion/move_joint
    # =========================================================
    def _robot_worker(self) -> None:
        while rclpy.ok() and (not self._stop.is_set()):
            try:
                seq, trash, bins = self._job_q.get(timeout=0.1)
            except queue.Empty:
                continue

            # If motion is disabled, just log and continue.
            if not self.enable_motion:
                self.get_logger().warn(
                    f"[robot] motion disabled (ENABLE_MOTION=0). Would move on seq={seq}. "
                    f"Set ENABLE_MOTION=1 to enable."
                )
                continue

            # Ensure move_joint service is ready
            if not self.movej_cli.wait_for_service(timeout_sec=2.0):
                self.get_logger().error(f"[robot] move_joint service not ready: {self.move_joint_srv}")
                continue

            # ---- Motion policy (minimal) ----
            # For now: if there is at least one trash item, run a single test movej.
            # You can replace this with a real pick-place plan later.
            if len(trash) == 0:
                self.get_logger().info(f"[robot] seq={seq}: no objects, skip motion")
                continue

            req = MoveJoint.Request()
            # pos is fixed-size array[6]; assign as list length 6
            req.pos = [float(v) for v in self.test_posj]
            req.vel = float(self.vel)
            req.acc = float(self.acc)
            req.time = float(self.time_s)
            req.radius = float(self.radius)
            req.mode = int(self.mode)
            req.blend_type = int(self.blend_type)
            req.sync_type = int(self.sync_type)

            self.get_logger().info(
                f"[robot] call move_joint | seq={seq} | pos={list(req.pos)} | vel={req.vel} acc={req.acc} time={req.time}"
            )

            fut = self.movej_cli.call_async(req)
            rclpy.spin_until_future_complete(self, fut, timeout_sec=self.motion_timeout_sec)

            if not fut.done():
                self.get_logger().error(f"[robot] move_joint timeout > {self.motion_timeout_sec:.1f}s (seq={seq})")
                continue

            try:
                resp: MoveJoint.Response = fut.result()
            except Exception as e:
                self.get_logger().error(f"[robot] move_joint call failed (seq={seq}): {e}")
                continue

            if not resp.success:
                self.get_logger().error(f"[robot] move_joint returned success=False (seq={seq})")
            else:
                self.get_logger().info(f"[robot] move_joint success=True (seq={seq})")

    def shutdown(self) -> None:
        self._stop.set()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ControlRunner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()