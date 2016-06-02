from rl.tools import Trajectory


class Trajectories:

    def __init__(self,observation_space=None,action_space=None):
        if (not observation_space is None):
            self.observation_space=observation_space
            self.action_space=action_space

        self.trajectories=[]

    def new_trajectory(self):
        self.push_trajectory(Trajectory())

    def push_trajectory(self,trajectory):
        self.trajectories.append(trajectory)

    def push_observation(self,o):
        self.trajectories[-1].push_observation(o)

    def push_action(self, o):
        self.trajectories[-1].push_action(o)

    def push_reward(self, o):
        self.trajectories[-1].push_reward(o)

    def size(self):
        return(len(self.trajectories))

    def clear(self):
        self.trajectories=[]