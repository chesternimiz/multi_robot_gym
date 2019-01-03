import rospy
from stage_launch import NeighborPosition
from nav_msgs import Odometry


class OdomReceiver:
    def __init__(self, index):
        self.index = index
        self.x = -1000
        self.y = 0
        self.z = 0
        self.sub = rospy.Subscriber("robot_"+str(self.index)+"/odom",Odometry, self.cb)

    def cb(self,msg):
        self



if __name__ == '__main__':
    rospy.init_node('stage_neighbor_position_sensor', anonymous=True)
    robot_num = rospy.get_param("robot_num", 10)
    subscribers = []
    p_x = []
    p_y = []
    for i in range(0,robot_num):
        p_x.append(0.)
        p_y.append(0.)
    for i in range(0,robot_num):
        subscribers.append( rospy.Subscriber("robot_"+str(i)+"/odom") )