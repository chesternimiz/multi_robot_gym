#!/usr/bin/env python

import gym
import numpy
from gym import wrappers
import rospy
import rospkg


import time
from distutils.dir_util import copy_tree
import os
import json
from multi_robot_gym.envs import gazebo_turtlebot_laser_maze0
from rl_algorithms.ppo.ppo_agent import PPOAgent
from gym.spaces import Box, Discrete

if __name__ == '__main__':

    rospy.init_node('ppo_example', anonymous=True)
    env = gym.make('GazeboTurtlebotLaserMaze0-v0')
    observation_space = Box(low=-numpy.inf, high=numpy.inf, shape=tuple([100]), dtype=numpy.float32)
    action_space = Box(low=-0.3, high=0.3, shape=tuple([1]), dtype=numpy.float32)
    #print action_space.shape
    #print observation_space.shape
    PPO = PPOAgent(action_space=action_space, observation_space=observation_space)
    #print PPO.a_ph.shape
    #print PPO.x_ph.shape

    last100Scores = [0] * 100
    last100ScoresIndex = 0
    last100Filled = False
    stepCounter = 0
    highest_reward = 0

    start_time = time.time()
    epochs = 1000
    steps = 1000
    current_epoch = 0

    #start iterating from 'current epoch'.
    for epoch in xrange(current_epoch+1, epochs+1, 1):
        observation, r, done, ep_ret, ep_len = env.reset(), 0, False, 0, 0
        observation = numpy.asarray(observation[0][0].ranges)
        #print str(observation)
        #print len(observation)
        episode_step = 0

        # run until env returns done
        for t in range(PPO.local_steps_per_epoch):
            # env.render()
            a, v_t, logp_t = PPO.get_action(observation)
            PPO.add_experience(observation, a[0], r, v_t, logp_t)
            env_action = [[[0.2, 0, 0, a[0][0]*0.3]]]
            # print env_action, a[0][0]*0.3
            newObservation, r, done, info = env.step(env_action)
            newObservation = numpy.asarray(newObservation[0][0].ranges)
            ep_ret += r
            ep_len += 1

            if done or episode_step >= steps:
                PPO.finish_path(r, done, newObservation)
                last100Scores[last100ScoresIndex] = episode_step
                last100ScoresIndex += 1
                if last100ScoresIndex >= 100:
                    last100Filled = True
                    last100ScoresIndex = 0
                if not last100Filled:
                    print ("EP " + str(epoch) + " - " + format(episode_step + 1) + "/" + str(steps) + " Episode steps ")
                else :
                    m, s = divmod(int(time.time() - start_time), 60)
                    h, m = divmod(m, 60)
                    print ("EP " + str(epoch) + " - " + format(episode_step + 1) + "/" + str(steps) + " Episode steps - last100 Steps : " + str((sum(last100Scores) / len(last100Scores))) + " - Cumulated R: " + str(ep_ret) + "  Time: %d:%02d:%02d" % (h, m, s))
                observation, r, done, ep_ret, ep_len = env.reset(), 0, False, 0, 0
                observation = numpy.asarray(observation[0][0].ranges)
                episode_step = 0
            episode_step += 1
            observation = newObservation
        print "PPO updating"
        PPO.update()

    env.close()