import rospy
import gym
import numpy as np
from gym.utils import seeding

class Simulator(gym.Env):
    def __init__(self,namespace=''):
        self.namespace=namespace
        self.robots=[]

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def pause_sim(self):
        raise NotImplementedError

    def unpause_sim(self):
        raise NotImplementedError

    def step(self, action):
        self.unpause_sim()
        for robot,act in self.robots,action:
