# Predictive Policy

# Predictive Recurrent Policy Gradient Tutorial

We explain here how the `PredictiveRecurrentPolicyGradient` policy can be used for making prediction (seen as a sequential process). It aims at targetting a muticlass classification problem where the sequential process acquires one features at each timestep as described for example in `Gabriel Dulac-Arnold, Ludovic Denoyer, Philippe Preux, Patrick Gallinari: Sequential approaches for learning datum-wise sparse representations. Machine Learning 89(1-2): 87-122 (2012)`. At the end of the acquisition process, the policy has to predict the category of the given example. 

First, we load a dataset and create the corresponding environment

```lua
local PROPORTION_TRAIN=0.5
local data,labels,nb_categories = unpack(rltorch.RLFile():read_libsvm('datasets/breast-cancer_scale'))

local parameters={}
parameters.training_examples, parameters.training_labels,parameters.testing_examples,parameters.testing_labels = unpack(rltorch.RLFile():split_train_test(data,labels,PROPORTION_TRAIN))

env = rltorch.SparseSequentialLearning_v0(parameters)
sensor=rltorch.BatchVectorSensor(env.observation_space)```

Second, we create the predictive policy. The classification problem is handled as a Negative Log-Lieklihood minimization problem.

```lua
local size_input=sensor:size()
local nb_actions=env.action_space.n
print("Size input = "..size_input)
print("Nb_actions = "..nb_actions)
print("Nb categories = "..nb_categories)

-- Creating the predictive policy
local N=10 -- the size of the latent space
local STDV=0.01
local initial_state=torch.Tensor(1,N):fill(0)

-- Creating the policy module which maps the latent space to the action space.
local module_policy=nn.Sequential():add(nn.Linear(N,nb_actions)):add(nn.SoftMax()):add(nn.ReinforceCategorical()); module_policy:reset(STDV)
local initial_recurrent_module = rltorch.RNN():rnn_cell(size_input,N,N); initial_recurrent_module:reset(STDV)
local recurrent_modules={}
for a=1,nb_actions do
  recurrent_modules[a]=rltorch.GRU():gru_cell(size_input,N)
  recurrent_modules[a]:reset(STDV)
end
--the module that maps the latent space to the final prediction
local predictive_module=nn.Linear(N,nb_categories); predictive_module:reset(STDV)

-- the predictive criterion
local criterion=nn.CrossEntropyCriterion()

local arguments={
    policy_module = module_policy,
    predictive_module=predictive_module,
    initial_state = initial_state,
    N = N,
    initial_recurrent_module = initial_recurrent_module,
    recurrent_modules = recurrent_modules,
    max_trajectory_size = NB_FEATURES+1,
    optim=optim.adam,
    optim_params= {
        learningRate =  0.001  
      },
    size_memory_for_bias=100,
    criterion=criterion
  }
  
policy=rltorch.PredictiveRecurrentPolicyGradient(env.observation_space,env.action_space,sensor,arguments)
```

At last, we train the policy, and evaluates it at each timestep over the testing set. Note that all the trajectories have the same size (NB_FEATURES) which corresponds to the number of features that will be acquired for each example to classify.

```lua
local train_losses={}
local test_losses={}
for i=1,NB_ITERATIONS do
    
    --- Evaluation on the test set
    policy.train=false
    local total_loss_test=0
    local observation,reward,done,feedback
      for j=1,TEST_SIZE do
      policy:new_episode(env:reset(true))      
      for t=1,NB_FEATURES do
        local action=policy:sample()      
        observation,reward,done,feedback=unpack(env:step(action)) 
        policy:observe(observation)
      end
      local prediction=policy:predict()
      local loss_test=criterion:forward(prediction,feedback.target)
      total_loss_test=total_loss_test+loss_test      
    end
    total_loss_test=total_loss_test/TEST_SIZE
    test_losses[i]=total_loss_test
    
    -- Evaluation + training on training examples
    policy.train=true
    local total_loss_train=0
    for j=1,TRAIN_SIZE do
      policy:new_episode(env:reset(false))      
      for t=1,NB_FEATURES do
        local action=policy:sample()      
        observation,reward,done,feedback=unpack(env:step(action)) 
        policy:observe(observation)
      end
      local prediction=policy:predict()
      local loss_train=criterion:forward(prediction,feedback.target)
      total_loss_train=total_loss_train+loss_train 
      policy:end_episode(feedback)
    end  
    total_loss_train=total_loss_train/TRAIN_SIZE
    train_losses[i]=total_loss_train
  
    if (i%1==0) then gnuplot.plot({"Training Loss",torch.Tensor(train_losses),"~"},{"Testing Loss",torch.Tensor(test_losses),"~"}) end
end
env:close()
```
