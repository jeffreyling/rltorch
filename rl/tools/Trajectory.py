#Describe a trajectory which is (o_1, a_1, o_2,a_2, ...,o_T) where T is the size of the trajectory.
#NB: the number of actions is T-1

class Trajectory:

    def __init__(self,observation_space=None,action_space=None):
        if (not observation_space is None):
            self.observation_space = observation_space
            self.action_space = action_space

        self._size=1
        self.observation_space=observation_space
        self.action_space=action_space
        self.observations=[]
        self.actions=[]
        self.rewards=[]

    def push_observation(self,observation):
        self.observations.append(observation)
        self._size = self._size + 1

    def push_action(self, action):
        self.actions.append(action)

    def push_reward(self,reward):
        self.rewards.append(reward)

    def size(self):
        return self._size




