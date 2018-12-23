import rospy


class RobotActuator:

    def __init__(self,name_space=''):  # declare publisher here
        self.name_space=name_space

    def act_once(self,action):  # publish action command here
        raise NotImplementedError
