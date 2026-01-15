from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple


@dataclass(frozen=True)
class TrashItem:
    type_id: int
    x: float
    y: float
    z: float
    angle: float


def parse_trash_flat(flat: List[float]) -> List[TrashItem]:
    if len(flat) % 5 != 0:
        raise ValueError(f"trash_list length must be multiple of 5, got {len(flat)}")

    out: List[TrashItem] = []
    for i in range(0, len(flat), 5):
        t, x, y, z, ang = flat[i:i+5]
        out.append(
            TrashItem(
                type_id=int(t),
                x=float(x),
                y=float(y),
                z=float(z),
                angle=float(ang),
            )
        )
    return out


def parse_bins(bins: List[List[float]]) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    for b in bins:
        if len(b) < 2:
            continue
        out.append((float(b[0]), float(b[1])))
    return out