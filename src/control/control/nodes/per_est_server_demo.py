import rclpy

from control.nodes.per_est_server import PerEstServer
from perception.utils import pipeline


def main(args=None):
    rclpy.init(args=args)
    node = PerEstServer()

    # Dummy estimation output: [type, x, y, z, angle] * N
    node.update_trash_list([
        1.0, 0.40, 0.20, 0.13, 60.0,
        2.0, 0.50, 0.25, 0.15, 30.0,
    ])

    # Dummy perception output: [x1, y1, x2, y2, ...]
    pipeline.processed_result["flat_clicked_xy"] = [300.0, 430.0, 500.0, 450.0]

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
