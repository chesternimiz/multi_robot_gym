from multi_robot_gym import robot_actuator
import rospy
from geometry_msgs.msg import Twist


class MoveBase(robot_actuator.RobotActuator):
    def __init__(self, namespace=''):
        robot_actuator.RobotActuator.__init__(self, namespace)
        if namespace != '':
            self.publisher = rospy.Publisher(namespace+'/cmd_vel', Twist, queue_size=10)
        else:
            self.publisher = rospy.Publisher('cmd_vel', Twist, queue_size=10) # remove / to use rosnode namespace to support multi envs in multi process
        self.constX = None   # support simple maze env, const linear speed

    def act_once(self, action):
        msg=Twist()
        if self.constX is None:
            msg.linear.x = action[0]
        else:
            msg.linear.x = self.constX
        msg.linear.y = action[1]
        msg.linear.z = action[2]
        msg.angular.z = action[3]


        self.publisher.publish(msg)
