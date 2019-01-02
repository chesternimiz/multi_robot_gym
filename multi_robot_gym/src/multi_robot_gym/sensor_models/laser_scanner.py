from multi_robot_gym import robot_sensor
from sensor_msgs.msg import LaserScan
import rospy


class LaserScanner(robot_sensor.RobotSensor):
    def __init__(self, namespace=''):
        robot_sensor.RobotSensor.__init__(self, topic="/scan", msgtype=LaserScan, namespace=namespace)






