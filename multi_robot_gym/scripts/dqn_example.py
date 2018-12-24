#!/usr/bin/env python

import rospy
import gym
from multi_robot_gym.envs import gazebo_turtlebot_laser_maze0

if __name__ == '__main__':

    rospy.init_node('dqn_example', anonymous=True)
    env = gym.make('GazeboTurtlebotLaserMaze0-v0')

    for episode in range(0, 500):
        observation = env.reset()
        done = False
        cumulated_R = 0
        while not done:
            action = [ [[0,0,0,0]] ]  # robot-actuator-actionDim
            observation, done, reward, info = env.step(action)
            cumulated_R += reward
            if done:
                print "ep"+str(episode+1)+"/500, cumulatedR="+str(cumulated_R)
                break
