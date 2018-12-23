import robot_actuator
import robot_sensor


class Robot:
    def __init__(self):
        self.actuators=[]
        self.sensors=[]

    def act_once(self, actions):
        for actuator, action in self.actuators, actions:
            actuator.act_once(action)

    def observe_once(self):
        observation = []
        for sensor in self.sensors:
            observation.append(sensor.wait_for_one_msg)   # TODO: parallel?
        return observation

    def get_last_ob(self):
        observation=[]
        for sensor in self.sensors:
            observation.append(sensor.data_queue[-1])
        return observation
