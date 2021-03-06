import rospy
import gym
import numpy as np
from gym.utils import seeding


class SimulatorEnv(gym.Env):
    def __init__(self,namespace=''):
        self.namespace = namespace
        self.robots = []

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def pause_sim(self):
        raise NotImplementedError

    def unpause_sim(self):
        raise NotImplementedError

    def reset_sim(self):
        raise NotImplementedError

    def ep_end(self, observation, action):
        raise NotImplementedError

    def get_reward(self, observation, action, done):
        raise NotImplementedError

    def get_info(self,observation):
        return []

    def step(self, action):   # action: [ [robot0_actuator_0, r0a1, r0a2...], [r1a0]...  ]
        self.unpause_sim()
        for i in range(len(self.robots)):
            self.robots[i].act_once(action[i])
        #observation = []
        #for robot in self.robots:
        #    observation.append(robot.observe_once())
        for robot in self.robots:
            robot.begin_observation()
        finish = False
        while not finish:
            finish = True
            for robot in self.robots:
                finish = (finish and robot.check_observation())
            rospy.rostime.wallsleep(0.01)
        self.pause_sim()
        observation = []
        for robot in self.robots:
            observation.append(robot.get_last_ob())

        done = self.ep_end(observation, action)
        reward = self.get_reward(observation, action, done)  # for envs where rewards are calculated from ground truth, define and add a ground truth sensor to robots
        info = self.get_info(observation)
        return observation, reward, done, info

    def reset(self):
        self.reset_sim()
        self.unpause_sim()
        observation = []
        for robot in self.robots:
            observation.append(robot.observe_once())
        self.pause_sim()
        return observation
