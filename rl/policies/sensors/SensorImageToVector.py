import numpy as np
from rl.policies.sensors import Sensor

class SensorImageToVector(Sensor):

    def __init__(self,observation_space):
        self.observation_space=observation_space
        self._size=1
        for t in range(len(self.observation_space.shape)):
            v=self.observation_space.shape[t]
            self._size=self._size*v


    def process(self,observation):
        return observation.astype(np.float).ravel()

    def size(self):
        return(self._size)