from multi_robot_gym.robot_models import turtlebot2_laser
from multi_robot_gym.simulators import gazebo_env
import rospy
from gym.envs.registration import register
from sensor_msgs.msg import LaserScan

register(
    id='GazeboTurtlebotLaserMaze0-v0',
    entry_point='multi_robot_gym:envs.gazebo_turtlebot_laser_maze0.GazeboTurtlebotLaserMaze0',
    timestep_limit=1000,
)


class GazeboTurtlebotLaserMaze0(gazebo_env.GazeboEnv):
    def __init__(self):
        super(GazeboTurtlebotLaserMaze0, self).__init__()
        self.robots.append(turtlebot2_laser.Turtlebot2Laser())
        self.robots[0].actuators[0].constX = 0.2
        self.cumulated_R = 0
        self.max_ang_speed = 0.3

    def reset_ops(self):
        self.cumulated_R = 0.
        self.step_passed = 0

    def ep_end(self, observation, action):
        episode_done = False
        laser_scan = observation[0][0]  #  robot 0 actuator 0
        for i, item in enumerate(laser_scan.ranges):
            if 0.2 > laser_scan.ranges[i] > 0:
                episode_done = True
        return episode_done

    def get_reward(self, observation, action):
        angular_speed = action[0][0][3]  # robot 0 actuator 0 action 3
        reward = round(15 * (self.max_ang_speed - abs(angular_speed) + 0.0335), 2)
        return reward
