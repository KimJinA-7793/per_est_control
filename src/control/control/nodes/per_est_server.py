from typing import List
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from perception.utils import pipeline
from rost_interfaces.msg import Float64Array
from rost_interfaces.srv import PerEstToControl


class PerEstServer(Node):
    def __init__(self):
        super().__init__('per_est_server')
        self.declare_parameter('trash_topic', '/estimation/pickup_commands')
        self._trash_topic = self.get_parameter('trash_topic').value

        self._trash_flat: List[float] = []
        self._bin_list: List[List[float]] = []

        self._sub_trash = self.create_subscription(Float32MultiArray, self._trash_topic, self._on_trash_msg, 10,)

        self._srv = self.create_service(PerEstToControl, 'per_est_to_control', self.on_request,)
        self.get_logger().info('PerEst Server ready: /per_est_to_control')

    # 쓰레기 데이터를 받아서 저장해주는 통로
    def update_trash_list(self, trash_list: List[float]) -> None:
        self._trash_flat = [float(v) for v in trash_list]

    # 쓰레기통 데이터를 받아서 저장해주는 통로
    def update_bin_list(self, bin_list: List[List[float]]) -> None:
        self._bin_list = [list(item) for item in bin_list]
    
    # 토픽으로 들어오는 결과를 msg.data로 _trash_flat 최신 값 업데이트
    def _on_trash_msg(self, msg: Float32MultiArray) -> None:
        self.update_trash_list(msg.data)

    # 클릭된 좌표 리스트를 bin_list 형태로 변환
    def bins_from_perception(self) -> None:
        flat_clicked_xy = pipeline.processed_result.get("flat_clicked_xy")
        if flat_clicked_xy is None:
            self._bin_list = []
            return
        if flat_clicked_xy and isinstance(flat_clicked_xy[0], (list, tuple)):
            self._bin_list = [[float(x), float(y)] for x, y in flat_clicked_xy]
            return
        self._bin_list = [
            [float(flat_clicked_xy[i]), float(flat_clicked_xy[i + 1])]
            for i in range(0, len(flat_clicked_xy), 2)
        ]

    # 받은 요청
    def on_request(self, request, response):
        self.bins_from_perception()
        response.trash_list = ([self._to_float64_array(self._trash_flat)] if self._trash_flat else [])
        response.bin_list = [self._to_float64_array(item) for item in self._bin_list]
        response.success = True
        response.message = 'Ok'
        return response
    
    # 파이썬 리스트를 ROS 메시지 형식으로 변환 및 msg.data에 넣기
    @staticmethod
    def _to_float64_array(data: List[float]) -> Float64Array:
        msg = Float64Array()
        msg.data = [float(v) for v in data]
        return msg
