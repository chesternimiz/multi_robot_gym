from multi_robot_gym import robot_sensor
from nav_msgs.msg import Odometry


class SelfSensor(robot_sensor.RobotSensor):
    def __init__(self, namespace=''):
        robot_sensor.RobotSensor.__init__(self, topic="base_pose_ground_truth", msgtype=Odometry, namespace=namespace)