from multi_robot_gym import simulator_env
import rospy
from std_srvs.srv import Empty


class StageEnv(simulator_env.SimulatorEnv):
    def __init__(self):
        super(StageEnv, self).__init__()
        self.reset_world = rospy.ServiceProxy('reset_positions', Empty)

    def pause_sim(self):  # stage ros doesn't support pause/unpause
        return

    def unpause_sim(self):
        return

    def reset_sim(self):
        self.reset_ops()
        rospy.wait_for_service('reset_positions')
        try:
            self.reset_world()
        except rospy.ServiceException as e:
            print ("reset_positions service call failed")

    def reset_ops(self):  # you may reset some variables here, e.g. cumulated rewards
        return []

    def ep_end(self, observation, action):
        raise NotImplementedError

    def get_reward(self, observation, action, done):
        raise NotImplementedError

    def get_info(self, observation):
        return []