#!/usr/bin/env python3
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Tuple

import rclpy

from control.nodes.control_client import ControlClient


# ----------------------------
# Data parsing helpers
# ----------------------------
@dataclass(frozen=True)
class TrashItem:
    type_id: int
    x: float
    y: float
    z: float
    angle: float


def parse_trash_flat(flat: List[float]) -> List[TrashItem]:
    """
    Input:
      flat: [type_id, x, y, z, angle] * N  (float list)

    Output:
      List[TrashItem] length N

    Failure path:
      - length % 5 != 0 -> ValueError
    """
    if len(flat) % 5 != 0:
        raise ValueError(f"trash_list length must be multiple of 5, got {len(flat)}")

    items: List[TrashItem] = []
    for i in range(0, len(flat), 5):
        t, x, y, z, ang = flat[i : i + 5]
        items.append(
            TrashItem(
                type_id=int(t),
                x=float(x),
                y=float(y),
                z=float(z),
                angle=float(ang),
            )
        )
    return items


def parse_bins(bin_list: List[List[float]]) -> List[Tuple[float, float]]:
    """
    Input:
      bin_list: [[x,y], [x,y], ...]

    Output:
      [(x,y), ...]
    """
    out: List[Tuple[float, float]] = []
    for b in bin_list:
        if not b or len(b) < 2:
            continue
        out.append((float(b[0]), float(b[1])))
    return out


# ----------------------------
# Demo main
# ----------------------------
def main(args=None) -> None:
    """
    Control Flow:
      - init -> create ControlClient
      - retry loop (max_wait_sec) calling request_data()
          - if trash_list empty => not READY yet -> sleep & retry
          - if non-empty => parse and print -> exit success
      - failure: timeout -> exit after logging

    Timing (정량):
      - retry_hz = 2 Hz (0.5s)
      - max_wait_sec = 15s (기본)
    """
    rclpy.init(args=args)
    node = ControlClient()

    retry_period_sec = 0.5  # 2 Hz
    max_wait_sec = 15.0
    deadline = time.time() + max_wait_sec

    try:
        while rclpy.ok() and time.time() < deadline:
            trash_list, bin_list = node.request_data()

            # request_data()가 PENDING이면 빈 리스트를 돌려주는 구조로 가정
            if not trash_list:
                node.get_logger().warn("not READY yet (trash_list empty) -> retry")
                time.sleep(retry_period_sec)
                continue

            # --- parse ---
            try:
                trash_items = parse_trash_flat(trash_list)
            except Exception as e:
                node.get_logger().error(f"parse_trash_flat failed: {e}")
                node.get_logger().info(f"raw trash_list={trash_list}")
                return

            bins = parse_bins(bin_list)

            # --- log ---
            node.get_logger().info(f"RAW trash_list={trash_list}")
            node.get_logger().info(f"RAW bin_list={bin_list}")
            node.get_logger().info(f"PARSED trash_items(N={len(trash_items)})={trash_items}")
            node.get_logger().info(f"PARSED bins(M={len(bins)})={bins}")

            # demo는 1회 성공하면 종료
            return

        node.get_logger().error(f"timeout: not READY within {max_wait_sec:.1f}s")

    finally:
        node.destroy_node()
        rclpy.shutdown()