import gym
import lutorpy as lua

require('rltorch') 
require('nn')
require('dpnn')
require('optim')


DISCOUNT_FACTOR=1.0
NB_TRAJECTORIES=10000
MAX_LENGTH=100

env = gym.make('CartPole-v0')
action_space=rltorch.Discrete(env.action_space.n)
observation_space=rltorch.Box(torch.fromNumpyArray(env.observation_space.low),torch.fromNumpyArray(env.observation_space.high))
sensor=rltorch.BatchVectorSensor(observation_space)
size_input=sensor._size()
print(size_input)
print("Inpust size is %d" % size_input)
nb_actions=env.action_space.n
print("Number of actions is %d " % nb_actions)

# Creating the policy module
module_policy=nn.Sequential()._add(nn.Linear(size_input,nb_actions))._add(nn.SoftMax())._add(nn.ReinforceCategorical())
module_policy._reset(0.001)


optim_params=lua.table(learningRate = 0.01)
arguments = lua.table(policy_module = module_policy, max_trajectory_size = MAX_LENGTH,   optim = optim.adam,  optim_params = optim_params, scaling_reward = 1.0/MAX_LENGTH,
 size_memory_for_bias = 100)
policy=rltorch.PolicyGradient(observation_space,action_space,sensor,arguments)

for i_episode in range(NB_TRAJECTORIES):
    observation = env.reset()
    print(observation)
    obs=torch.fromNumpyArray(observation)
    policy._new_episode(obs)
    total_reward=0
    current_discount=1.0
    for t in range(MAX_LENGTH):
        #env.render()
        action = policy._sample()
        action=action-1
        #env.action_space.sample()
        observation, reward, done, info = env.step(action)
        total_reward=total_reward+reward*current_discount
        current_discount=current_discount*DISCOUNT_FACTOR
        policy._observe(torch.fromNumpyArray(observation))
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            print("Reward is %f" % total_reward)
            break
    print("TReward = %f " % total_reward)
    policy._end_episode(total_reward)
    
