import rclpy
import DR_init
from rclpy.node import Node

from rost_interfaces.srv import PerEstToControl
from control.recycle import Recycle, ROBOT_ID


class ControlClient(Node):
    def __init__(self):
        super().__init__('control_client')
        self.client = self.create_client(PerEstToControl, 'per_est_to_control')

        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')

    def request_data(self):
        # request로 요청하고 future는 응답이 도착하면 채워질 객체 반환
        request = PerEstToControl.Request()
        future = self.client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        # 비동기 작업 실패 및 응답 없을시 처리
        if future.exception() is not None:
            self.get_logger().error(f'service call failed: {future.exception()}')
            return [], []

        response = future.result()
        if not response.success:
            self.get_logger().error(f'service returned failure: {response.message}')
            return [], []

        trash_list = [value for item in response.trash_list for value in list(item.data)]
        bin_list = [list(item.data) for item in response.bin_list]
        return trash_list, bin_list
