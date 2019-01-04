from multi_robot_gym import robot_sensor
from stage_launch.msg import NeighborInfo


class NeighborInfoReceiver(robot_sensor.RobotSensor):
    def __init__(self, namespace=''):
        robot_sensor.RobotSensor.__init__(self, topic="neighbor_info", msgtype=NeighborInfo, namespace=namespace)