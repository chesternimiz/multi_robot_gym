from multi_robot_gym.actuator_models import move_base
from multi_robot_gym.sensor_models import neighbor_info_receiver
from multi_robot_gym.sensor_models import self_sensor
from multi_robot_gym import robot


class SimpleSwarmRobot(robot.Robot):
    def __init__(self, robot_name=''):
        robot.Robot.__init__(self, robot_name)
        self.actuators.append(move_base.MoveBase(robot_name))
        self.sensors.append(neighbor_info_receiver.NeighborInfoReceiver(robot_name))
        self.sensors.append(self_sensor.SelfSensor(robot_name))
