--- This tutorial shows how a classical multiclass classification problem can be handled in a RL framework
--- In this tutorial, this is viewed as a classical RL problem

require ('optim') 
require('rltorch')
require('svm')

math.randomseed(os.time())


local NB_ITERATIONS=1000 -- The number of trajectories
local SIZE_ITERATION=1000 -- The number of training example to sample for each trajectory


local PROPORTION_TRAIN=0.5
local data,labels = unpack(rltorch.RLFile():read_libsvm('datasets/breast-cancer_scale'))
local parameters={}
parameters.training_examples, parameters.training_labels,parameters.testing_examples,parameters.testing_labels = unpack(rltorch.RLFile():split_train_test(data,labels,PROPORTION_TRAIN))

env = rltorch.MulticlassClassification_v0(parameters)
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
    size_memory_for_bias=100
  }
  
policy=rltorch.PolicyGradient(env.observation_space,env.action_space,sensor,arguments)

local train_rewards={}
local test_rewards={}
for i=1,NB_ITERATIONS do
    
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
    print("0/1 Reward at iteration "..i.." (train) is "..sum_reward)  
    if (i%1==0) then gnuplot.plot({"Training accuracy",torch.Tensor(train_rewards),"~"},{"Testing accuracy",torch.Tensor(test_rewards),"~"}) end
    
    policy:end_episode(sum_reward) -- The feedback provided for the whole episode here is the discounted sum of rewards      
end
env:close()


