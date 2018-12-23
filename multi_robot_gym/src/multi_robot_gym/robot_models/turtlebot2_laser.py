from multi_robot_gym.actuator_models import move_base
from multi_robot_gym.sensor_models import laser_scan
from multi_robot_gym import robot
import robot
import rospy


class Tutlebot2Laser(robot.Robot):
    def __init__(self,robot_name=''):
        super(Tutlebot2Laser,self).__init__(robot_name)
        self.actuators.append(move_base.MoveBase(robot_name))
        self.actuators.append(laser_scan.LaserScan(robot_name))


