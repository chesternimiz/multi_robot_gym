import gym
from multi_robot_gym.robot_models import simple_swarm_robot
from multi_robot_gym.simulators import stage_env
from multi_robot_gym.simulators import gazebo_env
from multi_robot_gym import simulator_env
import rospy
from gym.envs.registration import register
import numpy as np

register(
    id='StageSwarm5-v0',
    entry_point='multi_robot_gym:envs.stage_swarm_5.StageSwarmFive',
    #entry_point='multi_robot_gym:envs.gazebo_turtlebot_laser_maze0.GazeboTurtlebotLaserMaze0',
    timestep_limit=1000,
)


class StageSwarmFive(stage_env.StageEnv):
    def __init__(self):
        super(StageSwarmFive, self).__init__()
        self.robots.append(simple_swarm_robot.SimpleSwarmRobot(robot_name="robot_0/"))
        self.robots.append(simple_swarm_robot.SimpleSwarmRobot(robot_name="robot_1/"))
        self.robots.append(simple_swarm_robot.SimpleSwarmRobot(robot_name="robot_2/"))
        self.robots.append(simple_swarm_robot.SimpleSwarmRobot(robot_name="robot_3/"))
        self.robots.append(simple_swarm_robot.SimpleSwarmRobot(robot_name="robot_4/"))
        self.cumulated_R = 0.
        self.step_count = 0

    def reset_ops(self):
        self.cumulated_R = 0.
        self.step_count = 0

    def ep_end(self, observation, action):
        episode_done = False
        self.step_count += 1
        for i in range(0,5):
            odom_i = observation[i][1]
            x_i = odom_i.pose.pose.position.x
            y_i = odom_i.pose.pose.position.y
            for j in range(i+1,5):
                odom_j = observation[j][1]
                x_j = odom_j.pose.pose.position.x
                y_j = odom_j.pose.pose.position.y
                if np.square(x_i-x_j)+np.square(y_i-y_j) < np.square(0.5):
                    episode_done = True
                    print "episode end due to collision"
        if self.step_count >= 1000:
            episode_done = True
        return episode_done

    def get_reward(self, observation, action, done):
        reward = 0
        if done:
            if self.step_count < 1000:
                reward = -10000
        return reward

    def reset(self):
        return []