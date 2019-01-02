import rospy
import threading


class RobotSensor:

    def __init__(self, topic, msgtype, namespace=''):  # declare subscriber and sensor data queue here
        self.subscriber = None
        self.data_queue = []
        self.max_queue_length = 10
        self.namespace = namespace
        self.topic = topic
        self.msgtype = msgtype
        self.subscribe()
        self.msg_rcv_count = 0
        self.last_count = 0

    def subscribe(self):
        self.subscriber = rospy.Subscriber(self.namespace+self.topic, self.msgtype, self.cb)

    def cb(self, msg):
        if len(self.data_queue) >= self.max_queue_length:
            self.data_queue.pop(0)
        self.data_queue.append(msg)
        self.msg_rcv_count += 1

    def wait_new_msg(self):
        self.last_count = self.msg_rcv_count

    def check_new_msg(self):
        if self.last_count == self.msg_rcv_count:
            return False
        else:
            return True

    def get_last_msg(self):
        return self.data_queue[-1]

