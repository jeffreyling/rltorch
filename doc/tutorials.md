# Tutorials

## Tutorial 1 : Testing an environment
(see tutorials/t0_running.lua)

* Creation of a new environment: 

```lua
    env=rltorch.MountainCar_v0()
    print(env.observation_space)
    print(env.action_space)
```

* Sample 1 000 states from this environment choosing the actions with a uniform distribution!

```lua
    for i=1,1000 do
        env:render{mode="console"}
        local observation,reward,done,info=unpack(env:step(env.action_space:sample()))    
    end
```

Note that, in many environments, one can use `env:render{mode="qt",fps="30"}` if one wants to view the environment using Qt. `fps` is the max number of frames per second. In this code, the action is sampled from the `action_space` using the `Space:sample()` method

# Tutorial 2 : A (random) agent interacting with an environment
(see tutorials/t1_policy.lua)

*  Create the environment
```lua
    env = rltorch.MountainCar_v0()
```

* Create the agent (or policy)
```lua
    policy=rltorch.RandomPolicy(env.observation_space,env.action_space)
```

* Sample one trajectory:
  * First, sample the initial state, and transmit the initial observation to the agent

```lua
    local initial_observation=env:reset()
    policy:new_episode(env:reset())  
```
  * Second, the main loop:

```lua 
    local total_reward=0 
    while(true) do
      env:render{mode="console"}      
      --env:render{mode="qt"}      

      local action=policy:sample()  -- sample one action based on the policy
      local observation,reward,done,info=unpack(env:step(action))  -- apply the action
      policy:feedback(reward) -- transmit the (immediate) reward to the policy
      policy:observe(observation)      
      total_reward=total_reward+reward -- update the total reward for this trajectory

      --- Leave the loop if the environment is in a final state
      if (done) then
        break
      end
    end
    policy:end_episode(total_reward) -- Transmit the total trajectory reward to the policy
   env:close()
```

Note that, depending on the policy, sometimes the immediate feedback will be necessary (DeepQLearning for example), sometimes only the trajectory feedback will be enough (PolicyGradient)

# Tutorial 3 : Policy Gradient
(see tutorials/t2_policygradient)

We explain here how to use the REINFORCE algorithm on any environemnt (with discrete action space)

* Create the environment

```lua
    env = rltorch.CartPole_v0()
```
* Create a sensor. The sensor will be used here to transform the environment observations to a vector that will be used as a state features vector in the REINFORCE algorithm. The `BatchVectorSensor` used here just transforms the (n) torch.Tensor observation to a (1,n) torch.Tensor value (reshape)

```lua
   sensor=rltorch.BatchVectorSensor(env.observation_space)
```

* Get the size of the vectors and the number of actions
```lua
local size_input=sensor:size()
local nb_actions=env.action_space.n
```

* Build a dpnn module which will samples one action over the possible actions. The input of this module is the (1,n) vector provided by the sensor, and the output is a (1,A) onehot vector with a 1 for the chosen action. This module must implement the `reinforce` method provided in the `dpnn` package. The module used here is a linear module with a softmax and a multinomial sampling

```lua
local module_policy=nn.Sequential():add(nn.Linear(size_input,nb_actions)):add(nn.SoftMax()):add(nn.ReinforceCategorical())
module_policy:reset(0.01) --- initialize the values of the parameters
```

* Build the policy. 
  * First, one has to decide the parameters of this policy
```lua
local arguments={
    policy_module = module_policy,
```

  * For the `PolicyGradient` class, you must provide the maximum size of the trajectories (`MAX_LENGTH` here)

```lua
    max_trajectory_size = MAX_LENGTH,
```

  * You have to provide the optimization algorithm and its parameters:
```lua
    optim=optim.adam,
    optim_params= {
        learningRate =  0.01  
      },
```

  * The reward considered by the policy will be the original reward divided by 10 (just to provide an example)

```lua
    scaling_reward=0.1,
```

  * The average value of the last 100 trajectories will be used in reinforce to reduce the variance 
```lua 
    size_memory_for_bias=100
  }
```

* Now, we can build the policy
```lua
policy=rltorch.PolicyGradient(env.observation_space,env.action_space,sensor,arguments)
```

* And we can launch the learning loop. The `rewards` table is used to plot the rewards using `gnuplot`

```lua
for i=1,NB_TRAJECTORIES do
    policy:new_episode(env:reset())  
    local sum_reward=0.0
    
    for t=1,MAX_LENGTH do  
      env:render{mode="console"}      
      local action=policy:sample()      
      local observation,reward,done,info=unpack(env:step(action))    
      policy:feedback(reward) -- Not necessary when using PolicyGradient
      policy:observe(observation)      
      sum_reward=sum_reward+reward 
      if (done) then        
        break
      end
    end
    
    rewards[i]=sum_reward
    print("Reward at "..i.." is "..sum_reward)
    if (i%10==0) then gnuplot.plot(torch.Tensor(rewards),"|") end    
    policy:end_episode(sum_reward)
 end
env:close()
```

# Tutorial Deep Q-Learning (but the DeepQPolicy code must be checked...)

This is the same idea.... The only different is in the way the policy is built

```lua
env = rltorch.CartPole_v0()
sensor=rltorch.BatchVectorSensor(env.observation_space)

local size_input=sensor:size()
local nb_actions=env.action_space.n
```

* The module is here a simple `nn` module (no need to used `dpnn` stuffs)

```lua
module_policy=nn.Sequential():add(nn.Linear(size_input,size_input*2)):add(nn.Tanh()):add(nn.Linear(size_input*2,nb_actions))
module_policy:reset(0.01)

local arguments={
    policy_module = module_policy,
    max_trajectory_size = MAX_LENGTH,
    optim=optim.adam,
    optim_params= {
        learningRate =  0.01  
      },
    scaling_reward=1.0,
    size_minibatch=10,
    size_memory=100,
    discount_factor=1,
    epsilon_greedy=0.1
  }
  
policy=rltorch.DeepQPolicy(env.observation_space,env.action_space,sensor,arguments)
```

The following is the same. Note that, here, the feedback that is used is the feedback provided through the `policy:observe(...)` method. 

# Tutorial 4: The recurrent policy gradient

First, as usual, you have to build the environment, sensor, etc...

```lua
env = rltorch.CartPole_v0() 
math.randomseed(os.time())
sensor=rltorch.BatchVectorSensor(env.observation_space)

local size_input=sensor:size()
local nb_actions=env.action_space.n
```

Then, choose the size of the latent space
```lua
local N=10
```

Choose the value of the initial state

```lua
local initial_state=torch.Tensor(1,N):fill(0)
```

Build the three needed modules
* The module that samples an action (linear->softmax->multinomial) from the latent space

```lua
local module_policy=nn.Sequential():add(nn.Linear(N,nb_actions)):add(nn.SoftMax()):add(nn.ReinforceCategorical()); module_policy:reset(STDV)
```
* The module that computes the first latent space given the first observation of any new trajectory
```lua
local initial_recurrent_module = rltorch.RNN():rnn_cell(size_input,N,N); initial_recurrent_module:reset(STDV)
```

* The modules that computes the next state given the newly acquired observation (one module for each possible action)
```lua
local recurrent_modules={}
for a=1,nb_actions do
  recurrent_modules[a]=rltorch.GRU():gru_cell(size_input,N)
  recurrent_modules[a]:reset(STDV)
end
```

Then, we build the policy:
```lua
local arguments={
    policy_module = module_policy,
    initial_state = initial_state,
    N = N,
    initial_recurrent_module = initial_recurrent_module,
    recurrent_modules = recurrent_modules,
    max_trajectory_size = MAX_LENGTH,
    optim=optim.adam,
    optim_params= {
        learningRate =  0.001  
      },
    scaling_reward=1.0/MAX_LENGTH,
    size_memory_for_bias=100
  }
  
policy=rltorch.RecurrentPolicyGradient(env.observation_space,env.action_space,sensor,arguments)
```
And then, you can use this policy. WARNING, when facing very long trajectories, you will face some gradients problem (gradient clipping is not implemented in the policy for now)

# Tutorial 6 : Multiclass Classification as a 0/1 RL problem

Here we explain how a classical multiclass classification problem can be casted to a 0/1 reward problem: 

First, we load the dataset from a `libsvm` file

```lua
--- First: load the data from a libsvm files and create the right tensors
local PROPORTION_TRAIN=0.5
local data = svm.ascread('breast-cancer_scale')
local training_examples, training_labels,testing_examples,testing_labels = unpack(generateTrainTest(data,PROPORTION_TRAIN))
```

Next, we create the corresponding environment

```lua
local parameters={}
parameters.training_examples=training_examples
parameters.training_labels=training_labels
parameters.testing_examples=testing_examples
parameters.testing_labels=testing_labels
parameters.zero_one_reward=true

env = rltorch.MulticlassClassification_v0(parameters)
```

The followuing is very close to what we have made previously. The only difference is that we will evaluate the quality of the policy (classifier) on the testing examples at each iteration. 

Creation of the policy

```lua 
sensor=rltorch.BatchVectorSensor(env.observation_space)
--sensor=rltorch.TilingSensor2D(env.observation_space,30,30)

local size_input=sensor:size()
local nb_actions=env.action_space.n
print(env.action_space)
print(env.action_space.n)
print("Size input = "..size_input)
print("Nb_actions = "..nb_actions)
-- Creating the policy module
local module_policy=nn.Sequential():add(nn.Linear(size_input,nb_actions)):add(nn.SoftMax()):add(nn.ReinforceCategorical())
module_policy:reset(0.01)

local arguments={
    policy_module = module_policy,
    max_trajectory_size = SIZE_ITERATION,
    optim=optim.adam,
    optim_params= {
        learningRate =  0.01  
      },
    scaling_reward=1.0,
    size_memory_for_bias=100
  }
  
policy=rltorch.PolicyGradient(env.observation_space,env.action_space,sensor,arguments)
```

The main loop:

```lua
local train_rewards={}
local test_rewards={}
for i=1,NB_ITERATIONS do
```

*  Evaluation on the testing set (without learning the policy)
```lua
 --- Evaluation on the test set
    policy.train=false
    policy:new_episode(env:reset(true))  
    
    local sum_reward_test=0.0
    local flag=true 
    while(flag) do  
      env:render{mode="nothing"}      
      local action=policy:sample()      
      local observation,reward,done,info=unpack(env:step(action))    
      sum_reward_test=sum_reward_test+reward -- comptues the discounted sum of rewards
      --print(reward)
      if (done) then flag=false else policy:observe(observation) end       
    end
    test_rewards[i]=sum_reward_test/parameters.testing_examples:size(1)
    print("0/1 Reward at iteration "..i.." (test) is "..sum_reward_test)  
```

* Evaluation (and learning) over a sample of training examples

```lua
-- Evaluation + training on training examples
    policy.train=true
    policy:new_episode(env:reset())  
    local sum_reward=0.0
    
    for t=1,SIZE_ITERATION do  
      env:render{mode="nothing"}      
      local action=policy:sample()      
      local observation,reward,done,info=unpack(env:step(action))    
      policy:feedback(reward) -- the immediate reward is provided to the policy
      policy:observe(observation)      
      sum_reward=sum_reward+reward -- comptues the discounted sum of rewards
      if (done) then        
        break
      end
    end
    
    train_rewards[i]=sum_reward/SIZE_ITERATION
```
 Don't forget to close the loop... (see `t6_multiclass_classification.lua`)





