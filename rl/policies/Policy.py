from rl.policies import PolicyError

class Policy:
    def __init__(self,observation_space,action_space):
        self.observation_space=observation_space
        self.action_space=action_space

    def reset(self,initial_observation):
        pass

    def observe(self,observation):
        pass

    def feedback(self,reward):
        pass

    def sample(self):
        raise PolicyError("No sample implemented for this policy")

    def end(self):
        pass