from multi_robot_gym import simulator_env
import rospy
from std_srvs.srv import Empty


# For RL in gazebo, inherent this class and implement ep_end(self), get_reward(self) and add robots to self.robots
class GazeboEnv(simulator_env.SimulatorEnv):
    def __init__(self):
        super(GazeboEnv, self).__init__()
        self.unpause = rospy.ServiceProxy('gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('gazebo/pause_physics', Empty)
        self.reset_simulation = rospy.ServiceProxy('gazebo/reset_simulation', Empty)

    def pause_sim(self):
        rospy.wait_for_service('gazebo/pause_physics')
        try:
            self.pause()
        except rospy.ServiceException as e:
            print ("gazebo/pause_physics service call failed")

    def unpause_sim(self):
        rospy.wait_for_service('gazebo/unpause_physics')
        try:
            self.unpause()
        except rospy.ServiceException as e:
            print ("gazebo/unpause_physics service call failed")

    def reset_sim(self):
        self.reset_ops()
        rospy.wait_for_service('gazebo/reset_simulation')
        try:
            self.reset_simulation()
        except rospy.ServiceException as e:
            print ("gazebo/reset_simulation service call failed")

    def reset_ops(self):  # you may reset some variables here, e.g. cumulated rewards
        return []

    def ep_end(self, observation, action):
        raise NotImplementedError

    def get_reward(self, observation, action):
        raise NotImplementedError

    def get_info(self, observation):
        return []
