# realsense_loop.py
import pyrealsense2 as rs
import numpy as np
import cv2


def run(
    width=1280,
    height=720,
    fps=30,
    on_save=None,
    on_reset=None,
    on_click=None,
    update_depth_frame=None,
    update_color_image=None,
    get_points=None,
):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

    pipeline.start(config)

    align = rs.align(rs.stream.color)

    print("스페이스바: 이미지 저장 | r: 리셋 | esc: 종료")

    window_name = "RealSense Color"
    cv2.namedWindow(window_name)

    if on_click is not None:
        cv2.setMouseCallback(window_name, on_click)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())

            if update_depth_frame:
                update_depth_frame(depth_frame)
            if update_color_image:
                update_color_image(color_image)

            display_image = color_image.copy()

            if get_points:
                for x, y, _ in get_points():
                    cv2.circle(display_image, (int(x), int(y)), 5, (0, 0, 255), -1)

            cv2.imshow(window_name, display_image)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("r") and on_reset:
                on_reset()
            elif key == ord(" ") and on_save:
                on_save()
            elif key == 27:  # ESC
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("RealSense 종료")