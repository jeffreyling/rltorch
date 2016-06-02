from rl.policies import Policy


class DiscreteRandomPolicy(Policy):
    def __init__(self,observation_space,action_space):
        Policy.__init__(self,observation_space,action_space)

    def sample(self):
        return self.action_space.sample()
