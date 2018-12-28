import robot_actuator
import robot_sensor
import threading

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
            observation.append(sensor.wait_for_one_msg())   # TODO: parallel?
        return observation

    def get_last_ob(self):
        observation=[]
        for sensor in self.sensors:
            observation.append(sensor.data_queue[-1])
        return observation

    def observe_thread(self, sensor):
        return sensor.wait_for_one_msg()

