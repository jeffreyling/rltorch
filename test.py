import gym
import lutorpy as lua

require('rltorch') 
require('nn')
require('dpnn')

env = gym.make('CartPole-v0')
print(env.observation_space)
print(env.action_space)
action_space=rltorch.Discrete(env.action_space.n)
observation_space=rltorch.Box(torch.fromNumpyArray(env.observation_space.low),torch.fromNumpyArray(env.observation_space.high))
policy = rltorch.RandomPolicy(observation_space,action_space)
sensor=rltorch.BatchVectorSensor(observation_space)

size_input=sensor._size()
print(size_input)
print("Inpust size is %d" % size_input)
nb_actions=env.action_space.n
print("Number of actions is %d " % nb_actions)

# Creating the policy module
module_policy=nn.Sequential()._add(nn.Linear(size_input,size_input*2))._add(nn.Tanh())._add(nn.Linear(size_input*2,nb_actions))._add(nn.SoftMax())._add(nn.ReinforceCategorical())
#module_policy=nn.Sequential():add(nn.Linear(size_input,nb_actions)):add(nn.SoftMax()):add(nn.ReinforceCategorical())
module_policy._reset(0.001)


for i_episode in range(20):
    observation = env.reset()
    policy._new_episode(torch.fromNumpyArray(observation))
    for t in range(100):
        env.render()
        print(observation)
        action = policy._sample()
        action=action-1
        #env.action_space.sample()
        observation, reward, done, info = env.step(action)
        policy._observe(torch.fromNumpyArray(observation))
        
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
    policy._end_episode()
    
