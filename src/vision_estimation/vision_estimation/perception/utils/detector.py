# detector.py
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
import os

from ament_index_python.packages import get_package_share_directory
from .depth_utils import box_center_pixel


def _get_calib_path() -> str:
    pkg_share = get_package_share_directory("vision_estimation")
    return os.path.join(
        pkg_share,
        "vision_estimation",
        "perception",
        "resources",
        "camcalib.npz",
    )


CALIB_PATH = _get_calib_path()


def long_side_angle_0_180(rect) -> float:
    box = cv2.boxPoints(rect).astype(np.float32)
    edges = [box[(i + 1) % 4] - box[i] for i in range(4)]
    lens = [float(np.hypot(v[0], v[1])) for v in edges]
    v = edges[int(np.argmax(lens))]
    ang = float(np.degrees(np.arctan2(v[1], v[0])))
    if ang < 0:
        ang += 180.0
    return ang


def suppress_blue_boxes(blue_boxes, green_polys):
    filtered = []
    for bx1, by1, bx2, by2 in blue_boxes:
        cx = (bx1 + bx2) // 2
        cy = (by1 + by2) // 2
        keep = True
        for poly in green_polys:
            if cv2.pointPolygonTest(poly, (cx, cy), False) >= 0:
                keep = False
                break
        if keep:
            filtered.append((bx1, by1, bx2, by2))
    return filtered


class DepthDBSCANVisualizer:
    def __init__(self, calib_path: str = CALIB_PATH):
        self.color = None
        self.depth = None

        calib = np.load(calib_path)
        self.T_cam_to_work = calib["T_cam_to_work"]
        self.camera_matrix = calib["camera_matrix"]

        self.fx = float(self.camera_matrix[0, 0])
        self.fy = float(self.camera_matrix[1, 1])
        self.cx = float(self.camera_matrix[0, 2])
        self.cy = float(self.camera_matrix[1, 2])

        self.depth_scale = 0.001
        self.Z_floor = 1.5

        self.roi = (450, 160, 820, 440)
        self.MIN_GREEN_BOX_AREA = 250
        self.MIN_GREEN_BOX_EDGE = 20

        print("ROI + Depth + DBSCAN + Small Green Box Filter Ready")

    # ---- 이하 코드는 기존 그대로 ----
    # update(), extract_objects_dbscan_rotated(), extract_transparent_rgb(), run()


    def update(self, color, depth):
        self.color = color
        self.depth = depth

    def in_roi(self, u, v):
        x1, y1, x2, y2 = self.roi
        return x1 <= u <= x2 and y1 <= v <= y2

    def pixel_to_world(self, u, v, depth_cm):
        Xc = (u - self.cx) * depth_cm / self.fx
        Yc = (v - self.cy) * depth_cm / self.fy
        Zc = depth_cm

        Pc = np.array([Xc, Yc, Zc, 1.0], dtype=np.float64)
        Pw = self.T_cam_to_work @ Pc

        # 실환경 보정값 (하드코딩 유지)
        Pw[2] = -Pw[2]

        return Pw[:3]

    def extract_objects_dbscan_rotated(self, eps=2.0, min_samples=50):
        if self.depth is None:
            return []

        h, w = self.depth.shape
        points_2d, points_world = [], []

        for v in range(0, h, 2):
            for u in range(0, w, 2):
                if not self.in_roi(u, v):
                    continue

                d = self.depth[v, u]
                if d <= 0:
                    continue

                depth_cm = float(d) * self.depth_scale * 100.0
                Pw = self.pixel_to_world(u, v, depth_cm)

                if Pw[2] >= self.Z_floor:
                    points_2d.append([u, v])
                    points_world.append(Pw)

        if not points_world:
            return []

        labels = DBSCAN(eps=eps, min_samples=min_samples).fit(points_world).labels_
        green_items = []

        for label in set(labels):
            if label == -1:
                continue

            idx = np.where(labels == label)[0]
            pixels = np.array([points_2d[i] for i in idx], dtype=np.float32)
            if len(pixels) < 20:
                continue

            rect = cv2.minAreaRect(pixels)
            w_rect, h_rect = rect[1]

            if w_rect * h_rect < self.MIN_GREEN_BOX_AREA:
                continue
            if min(w_rect, h_rect) < self.MIN_GREEN_BOX_EDGE:
                continue

            angle = long_side_angle_0_180(rect)
            box = cv2.boxPoints(rect).astype(int)

            green_items.append({"box": box, "angle": angle})

        return green_items

    def extract_transparent_rgb(self):
        if self.color is None or self.depth is None:
            return []

        gray = cv2.cvtColor(self.color, cv2.COLOR_BGR2GRAY)

        bright = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            31, -5
        )

        edges1 = cv2.Canny(gray, 20, 60)
        edges2 = cv2.Canny(gray, 50, 150)
        edges = cv2.bitwise_or(edges1, edges2)
        rgb_mask = cv2.bitwise_or(bright, edges)

        depth = self.depth.astype(np.float32)
        depth_blur = cv2.medianBlur(depth, 5)
        grad = np.abs(cv2.Laplacian(depth_blur, cv2.CV_32F))

        depth_hole = np.zeros_like(depth, dtype=np.uint8)
        depth_hole[(depth == 0) | (grad > 20)] = 255

        mask = cv2.bitwise_and(rgb_mask, depth_hole)

        x1, y1, x2, y2 = self.roi
        roi_mask = np.zeros_like(mask)
        roi_mask[y1:y2, x1:x2] = 255
        mask = cv2.bitwise_and(mask, roi_mask)

        mask = cv2.erode(mask, np.ones((1, 1), np.uint8), iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 3100 :
                continue

            cnt_mask = np.zeros_like(mask)
            cv2.drawContours(cnt_mask, [cnt], -1, 255, -1)

            dist = cv2.distanceTransform(cnt_mask, cv2.DIST_L2, 5)
            _, _, _, max_loc = cv2.minMaxLoc(dist)
            cx, cy = max_loc

            box_size = int(np.sqrt(area) * 0.6)

            x1b = max(cx - box_size // 2, 0)
            y1b = max(cy - box_size // 2, 0)
            x2b = min(cx + box_size // 2, mask.shape[1])
            y2b = min(cy + box_size // 2, mask.shape[0])

            boxes.append((x1b, y1b, x2b, y2b))

        return boxes

    def run(self):
        if self.color is None or self.depth is None:
            raise RuntimeError("run() 전에 update(color, depth)를 먼저 호출하세요.")

        vis = self.color.copy()

        x1, y1, x2, y2 = self.roi
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)

        green_items = self.extract_objects_dbscan_rotated()
        green_polys = [it["box"] for it in green_items]

        blue_boxes = self.extract_transparent_rgb()
        blue_boxes = suppress_blue_boxes(blue_boxes, green_polys)

        items = []
        idx = 0

        for it in green_items:
            poly = it["box"]
            ang = float(it["angle"])
            items.append({
                "id": idx,
                "type": "green",
                "poly": poly,
                "angle": ang,
                "rect": None,
            })
            idx += 1

        for (x1b, y1b, x2b, y2b) in blue_boxes:
            poly = np.array(
                [[x1b, y1b], [x2b, y1b], [x2b, y2b], [x1b, y2b]],
                dtype=np.int32
            )
            items.append({
                "id": idx,
                "type": "blue",
                "poly": poly,
                "angle": 0.0,
                "rect": (int(x1b), int(y1b), int(x2b), int(y2b)),
            })
            idx += 1

        for item in items:
            color = (0, 255, 0) if item["type"] == "green" else (255, 0, 0)
            box = item["poly"]
            cv2.drawContours(vis, [box], 0, color, 2)

            cx, cy = box_center_pixel(box)
            cv2.circle(vis, (cx, cy), 5, (0, 255, 0), -1)

            text_y = int(np.min(box[:, 1]) - 10)
            cv2.putText(
                vis, f"{item['id']}",
                (cx, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
            )

        return vis, items


if __name__ == "__main__":
    raise SystemExit("이 파일은 단독 실행용이 아닙니다. main.py에서 import 해서 사용하세요.")