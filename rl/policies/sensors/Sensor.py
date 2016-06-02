#The gol of a sensor is to produce an information that can be used by a policy
from rl.policies.sensors.SensorError import SensorError

class Sensor:

    def process(self,observation):
        raise SensorError("No process function defined")

    def size(self):
        raise SensorError("No size defined")
