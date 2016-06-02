from rl.policies import Policy
from rl.tools import Trajectories
from rl.tools import Trajectory


class MemoryPolicy(Policy):

    def __init__(self,policy):
        Policy.__init__(self,policy.observation_space,policy.action_space)

        self.policy=policy
        self.trajectories=Trajectories(self.observation_space,self.action_space)

    def reset(self, initial_observation):
        self.trajectories.new_trajectory()
        self.trajectories.push_observation(initial_observation)
        self.policy.reset(initial_observation)

    def observe(self, observation):
        self.trajectories.push_observation(observation)
        self.policy.observe(observation)

    def feedback(self,reward):
        self.trajectories.push_reward(reward)
        self.policy.feedback(reward)

    def sample(self):
        action=self.policy.sample()
        self.trajectories.push_action(action)
        return action

    def clear_memory(self):
        self.trajectories = Trajectories(self.observation_space, self.action_space)

