from multi_robot_gym.actuator_models import move_base
from multi_robot_gym.sensor_models import laser_scanner
from multi_robot_gym import robot
import rospy
from gym.spaces import Box, Discrete
import numpy as np

class Turtlebot2Laser(robot.Robot):
    def __init__(self, robot_name='',constlinearspeed=False):
        robot.Robot.__init__(self, robot_name)
        self.actuators.append(move_base.MoveBase(robot_name))
        self.sensors.append(laser_scanner.LaserScanner(robot_name))
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(1, 100), dtype=np.float32)
        if constlinearspeed:
            self.action_space = Box(low=-0.3, high=0.3, shape=(1, 1), dtype=np.float32)  # angular.z
        else:
            self.action_space = Box(low=-5,high=5,shape=(1,2),dtype=np.float32)  # linear.x angular.z



