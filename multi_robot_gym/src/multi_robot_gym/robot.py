import robot_actuator
import robot_sensor
import rospy

class Robot:
    def __init__(self,robot_name=''):
        self.actuators=[]
        self.sensors=[]
        self.robot_name=robot_name

    def act_once(self, actions):
        for i in range(len(self.actuators)):
            self.actuators[i].act_once(actions[i])

    def observe_once(self):
        observation = []
        for sensor in self.sensors:
            sensor.wait_new_msg()
        finish = False
        while not finish:
            rospy.rostime.wallsleep(0.01)
            finish = True
            for sensor in self.sensors:
                finish = finish and sensor.check_new_msg()
        for sensor in self.sensors:
            observation.append(sensor.get_last_msg())
        return observation

    def get_last_ob(self):
        observation=[]
        for sensor in self.sensors:
            observation.append(sensor.get_last_msg())
        return observation

    def begin_observation(self):
        for sensor in self.sensors:
            sensor.wait_new_msg()

    def check_observation(self):
        for sensor in self.sensors:
            if sensor.check_new_msg() == False:
                return False
        return True

