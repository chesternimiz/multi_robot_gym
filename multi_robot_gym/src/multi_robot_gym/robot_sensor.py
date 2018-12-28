import rospy
import threading


class RobotSensor():

    def __init__(self, namespace=''):  # declare subscriber and sensor data queue here
        self.subscriber = None
        self.data_queue = []
        self.namespace = namespace

    def wait_for_one_msg(self):
        raise NotImplementedError

    def subscribe(self):  # subscribe topic here
        raise NotImplementedError
