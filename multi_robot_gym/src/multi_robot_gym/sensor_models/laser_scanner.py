from multi_robot_gym import robot_sensor
from sensor_msgs.msg import LaserScan
import rospy


class LaserScanner(robot_sensor.RobotSensor):
    def __init__(self, namespace=''):
        robot_sensor.RobotSensor.__init__(self, namespace)

    def wait_for_one_msg(self):
        data = None
        while data is None and not rospy.is_shutdown():
            try:
                data = rospy.wait_for_message(self.namespace+"/scan", LaserScan, timeout=5.0)
            except:
                rospy.logerr("Current "+self.namespace+"/scan not ready yet, retrying for getting laser_scan")
        # self.data_queue.append(data) #Don't do this because we recommend data recording via subscription
        return data

    def subscribe(self):  # subscribe topic here
        self.subscriber = rospy.Subscriber(self.namespace+"/scan", LaserScan, self.cb)

    def cb(self, msg):
        self.data_queue.append(msg)

