--- This tutorial shows how a classical multiclass classification problem can be handled in a RL framework
--- In this tutorial, this is viewed as a classical RL problem

require ('optim') 
require('rltorch')
require('svm')

function generateTrainTest(data,ptrain)
  local nb_examples=#data
  local train_or_test={}
  local index_categories={}
  local nb_cat=0
  local max_feature_index=0
  local nb_train=0
  local nb_test=0
  for i=1,nb_examples do
    if (math.random()<ptrain) then train_or_test[i]="train" nb_train=nb_train+1 else train_or_test[i]="test"; nb_test=nb_test+1 end
    local vf,mf=data[i][2][1]:max(1)
    if (vf[1]>max_feature_index) then max_feature_index=vf[1] end
    
    local cat=data[i][1]
    if (index_categories[cat]==nil) then
      index_categories[cat]=nb_cat+1 
      nb_cat=nb_cat+1
    end    
  end
  
  local training_examples=torch.Tensor(nb_train,max_feature_index):fill(0)
  local training_labels=torch.Tensor(nb_train,1):fill(0)
  local testing_examples=torch.Tensor(nb_test,max_feature_index):fill(0)
  local testing_labels=torch.Tensor(nb_test,1):fill(0)
  
  local pos_train=1
  local pos_test=1
  for i=1,nb_examples do
    local i_f=data[i][2][1]
    local v_f=data[i][2][2]
    if (train_or_test[i]=="train") then
      for k=1,i_f:size(1) do training_examples[pos_train][i_f[k]]=v_f[k] end
      training_labels[pos_train]=index_categories[data[i][1]]
      pos_train=pos_train+1
    else
      for k=1,i_f:size(1) do testing_examples[pos_test][i_f[k]]=v_f[k] end
      testing_labels[pos_test]=index_categories[data[i][1]]
      pos_test=pos_test+1
    end
  end
  print("Number of categories is "..nb_cat)   
  return {training_examples,training_labels,testing_examples,testing_labels}   
end

math.randomseed(os.time())


local NB_ITERATIONS=1000 -- The number of trajectories
local SIZE_ITERATION=1000 -- The number of training example to sample for each trajectory


--- First: load the data from a libsvm files and create the right tensors
local PROPORTION_TRAIN=0.5
local data = svm.ascread('datasets/breast-cancer_scale')
local training_examples, training_labels,testing_examples,testing_labels = unpack(generateTrainTest(data,PROPORTION_TRAIN))

local parameters={}
parameters.training_examples=training_examples
parameters.training_labels=training_labels
parameters.testing_examples=testing_examples
parameters.testing_labels=testing_labels
parameters.zero_one_reward=true

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
    scaling_reward=1.0,
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


