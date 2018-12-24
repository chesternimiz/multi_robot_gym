from multi_robot_gym.actuator_models import move_base
from multi_robot_gym.sensor_models import laser_scanner
from multi_robot_gym import robot
import rospy


class Turtlebot2Laser(robot.Robot):
    def __init__(self, robot_name=''):
        robot.Robot.__init__(self, robot_name)
        self.actuators.append(move_base.MoveBase(robot_name))
        self.sensors.append(laser_scanner.LaserScanner(robot_name))


