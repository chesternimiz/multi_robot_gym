from multi_robot_gym import robot_actuator
#import robot_actuator
import rospy
from geometry_msgs.msg import Twist

class MoveBase(robot_actuator.RobotActuator):
    def __init__(self,namespace=''):
        super(MoveBase,self).__init__(namespace)
        self.publisher = rospy.Publisher(namespace+'/cmd_cel',Twist,queue_size=10)

    def act_once(self,action):
        msg=Twist()
        msg.linear.x=action[0]
        msg.linear.y=action[1]
        msg.linear.z=action[2]
        msg.angular.z=action[3]
        self.publisher.publish(msg)