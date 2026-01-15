# pipeline.py
import os
import numpy as np
import cv2

from . import click_points
from .detector import DepthDBSCANVisualizer
from . import depth_utils
from .coordinate import Coordinate
from .settings import OUTPUT_DIR, COLOR_PATH, DEPTH_PATH

detector = DepthDBSCANVisualizer()

color_image = None
depth_image = None
points_3d = []

processed_result = {
    "color": None,
    "depth": None,
    "points_3d": None,

    "vis": None,
    "items": None,                 # ✅ 추가: detector raw items
    "boxes": None,                 # poly list (기존 유지)
    "bboxes_xyxy_sorted": None,    # ✅ 추가: (id,x1,y1,x2,y2,type) id정렬

    "green_items": None,
    "world_list": None,
    "clicked_world_xy_list": None,
    "flat_clicked_xy": None,
    "flat_world_list": None,
}


def _poly_to_xyxy(poly) -> tuple[int, int, int, int]:
    """poly: (4,2) or list of points -> bbox(x1,y1,x2,y2)"""
    pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
    x1 = int(np.floor(np.min(pts[:, 0])))
    y1 = int(np.floor(np.min(pts[:, 1])))
    x2 = int(np.ceil(np.max(pts[:, 0])))
    y2 = int(np.ceil(np.max(pts[:, 1])))
    return x1, y1, x2, y2


def _rect_to_xyxy(rect) -> tuple[int, int, int, int]:
    """
    rect가 무엇이든 bbox로 통일:
      - (x1,y1,x2,y2)
      - (cx,cy,w,h,angle) / cv2.minAreaRect
      - 기타: boxPoints로 처리 시도
    """
    try:
        # (x1,y1,x2,y2) 형태로 들어오는 경우
        if isinstance(rect, (list, tuple)) and len(rect) == 4 and all(isinstance(v, (int, float)) for v in rect):
            x1, y1, x2, y2 = rect
            return int(x1), int(y1), int(x2), int(y2)

        # cv2.minAreaRect: ((cx,cy),(w,h),angle)
        if isinstance(rect, (list, tuple)) and len(rect) == 3:
            box = cv2.boxPoints(rect)  # (4,2)
            return _poly_to_xyxy(box)

        # (cx,cy,w,h,angle) 형태
        if isinstance(rect, (list, tuple)) and len(rect) == 5:
            cx, cy, w, h, ang = rect
            r = ((float(cx), float(cy)), (float(w), float(h)), float(ang))
            box = cv2.boxPoints(r)
            return _poly_to_xyxy(box)

    except Exception:
        pass

    # 최후: 실패 시 0 bbox
    return 0, 0, 0, 0


def save_cam():
    """
    return:
      processed_result (dict)
      keys:
        color, depth, points_3d, vis, items, boxes,
        bboxes_xyxy_sorted, world_list,
        clicked_world_xy_list, flat_clicked_xy, flat_world_list
    """
    global color_image, depth_image, points_3d, processed_result

    # 1) 스냅샷 + 클릭포인트 3D 계산
    color, depth, points = click_points.Save_Cam()
    color_image = color
    depth_image = depth
    points_3d = points  # [(X,Y,Z), ...] (클릭 기반 world)

    processed_result["color"] = color
    processed_result["depth"] = depth
    processed_result["points_3d"] = points_3d

    if color is None or depth is None:
        print("save_cam: color/depth 없음")
        return processed_result

    # 1-1) 클릭 world에서 XY만 뽑아 저장
    clicked_world_xy_list = [[float(p[0]), float(p[1])] for p in points_3d]
    flat_clicked_xy = clicked_world_xy_list

    # 2) detect 실행
    detector.update(color, depth)
    vis, items = detector.run()

    # 3) 저장 (디버그용 vis 저장)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cv2.imwrite(COLOR_PATH, vis)  # vis 저장(기존 유지)
    np.save(DEPTH_PATH, depth)

    # ✅ items 저장
    processed_result["items"] = items

    # poly boxes (기존 유지)
    boxes = [it["poly"] for it in items]
    processed_result["boxes"] = boxes

    # ✅ id 정렬 bbox 생성: (id,x1,y1,x2,y2,type)
    bbox_rows = []
    for it in items:
        obj_id = int(it["id"])
        obj_type = it.get("type", "unknown")

        if obj_type == "green":
            x1, y1, x2, y2 = _poly_to_xyxy(it["poly"])
        else:
            # blue는 rect 기반 bbox
            x1, y1, x2, y2 = _rect_to_xyxy(it.get("rect", None))

        bbox_rows.append((obj_id, x1, y1, x2, y2, obj_type))

    bbox_rows = sorted(bbox_rows, key=lambda r: r[0])
    processed_result["bboxes_xyxy_sorted"] = bbox_rows

    # 4) world_list 생성 (id 유지)
    fake_depth = depth_utils.FakeDepthFrameFromNpy(depth)
    coord = Coordinate()

    world_list = []
    for it in items:
        obj_id = int(it["id"])
        obj_type = it["type"]

        Pw = None
        if obj_type == "green":
            cx, cy = depth_utils.box_center_pixel(it["poly"])
            Pw = coord.pixel_to_world(cx, cy, fake_depth)
            ang = float(it["angle"])
        else:
            Pw = depth_utils.blue_rect_to_world_safe(
                it["rect"], depth, search_step=2
            )
            ang = 0.0

        if Pw is None:
            X = Y = Z = 0.0
        else:
            X, Y, Z = map(float, Pw[:3])

        world_list.append({
            "id": obj_id,
            "type": obj_type,
            "world": (float(X), float(Y), float(Z)),
            "angle": float(ang),
        })

    # 4-1) flat_world_list (ID 순서 유지)
    flat_world_list = []
    for it in sorted(world_list, key=lambda d: d["id"]):
        X, Y, Z = it["world"]
        flat_world_list.extend([it["id"], X, Y, Z, float(it["angle"])])

    processed_result["vis"] = vis
    processed_result["world_list"] = world_list
    processed_result["clicked_world_xy_list"] = clicked_world_xy_list
    processed_result["flat_clicked_xy"] = flat_clicked_xy
    processed_result["flat_world_list"] = flat_world_list
    processed_result["green_items"] = None

    if boxes:
        cv2.imshow("Detect Result", vis)
        cv2.waitKey(1)

    print("flat_world_list:", flat_world_list)
    print("flat_clicked_xy:", flat_clicked_xy)
    print("bboxes_xyxy_sorted:", bbox_rows)

    return processed_result