#!/usr/bin/env python

import rospy
import gym
from multi_robot_gym.envs import stage_swarm_5
from multi_robot_gym.envs import gazebo_turtlebot_laser_maze0

if __name__ == '__main__':

    rospy.init_node('simple_example', anonymous=True)
    env = gym.make('StageSwarm5-v0')

    for episode in range(0, 500):
        observation = env.reset()

        rate = rospy.Rate(100)
        done = False
        cumulated_R = 0
        while not done:
            one_robot_action = [[0.5, 0, 0, 0]]
            action=[]
            for i in range(0, 5):
                action.append(one_robot_action)
            observation, reward, done, info = env.step(action)
            laser_msg = observation[0][0]   # robot0 actuator0
            # print laser_msg
            cumulated_R += reward
            rate.sleep()  # replace with learning algorithm
            if done:
                print "ep"+str(episode+1)+"/500, cumulatedR="+str(cumulated_R)
                break