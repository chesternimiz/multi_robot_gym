#!/usr/bin/env python

import rospy
from stage_launch.msg import NeighborInfo
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point
from geometry_msgs.msg import Quaternion
from geometry_msgs.msg import Vector3
import numpy


class OdomReceiver:
    def __init__(self, index):
        self.index = index
        self.x = -1000
        self.y = 0
        self.z = 0
        self.vx = 0
        self.vy = 0
        self.vz = 0
        self.az = 0
        self.sub = rospy.Subscriber("robot_"+str(self.index)+"/base_pose_ground_truth", Odometry, self.cb)
        self.pub = rospy.Publisher("robot_"+str(self.index)+"/neighbor_info", NeighborInfo,queue_size=10)

    def cb(self,msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        self.z = msg.pose.pose.position.z
        self.vx = msg.twist.twist.linear.x
        self.vy = msg.twist.twist.linear.y
        self.vz = msg.twist.twist.linear.z
        self.az = msg.twist.twist.angular.z



if __name__ == '__main__':
    rospy.init_node('stage_neighbor_position_sensor', anonymous=True)
    robot_num = rospy.get_param("robot_num", 10)
    R = rospy.get_param("R",10.)
    odom_receivers = []
    for i in range(0, robot_num):
        odom_receivers.append(OdomReceiver(i))

    is_neighbor = []
    for i in range(0,robot_num):
        is_neighbor.append([])
        for j in range (0,robot_num):
            is_neighbor[i].append(False)
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        for i in range(0,robot_num):
            for j in range(i+1,robot_num):
                if numpy.square((odom_receivers[i].x - odom_receivers[j].x))+numpy.square((odom_receivers[i].y - odom_receivers[j].y)) <= numpy.square(R):
                    is_neighbor[i][j] = True
                    is_neighbor[j][i] = True
                else:
                    is_neighbor[i][j] = False
                    is_neighbor[j][i] = False
        for i in range(0,robot_num):
            msg = NeighborInfo()
            msg.neighbor_num = 0
            for j in range(0,robot_num):
                if is_neighbor[i][j]:
                    msg.positions.append(Point())
                    msg.quaternions.append(Quaternion())
                    msg.linear_vels.append(Vector3())
                    msg.angular_vels.append(Vector3())
                    msg.positions[msg.neighbor_num].x = odom_receivers[j].x
                    msg.positions[msg.neighbor_num].y = odom_receivers[j].y
                    msg.linear_vels[msg.neighbor_num].x = odom_receivers[j].vx
                    msg.linear_vels[msg.neighbor_num].y = odom_receivers[j].vy  #TODO: support 2D and linear vel only
                    msg.neighbor_num += 1
            odom_receivers[i].pub.publish(msg)
        rate.sleep()



