require ('optim') 
require('rltorch')
log=rltorch.ExperimentLogConsole()
--log=rltorch.ExperimentLogCSV(false,"tmp/","now")

env = rltorch.MountainCar_v0()
env=rltorch.MonitoredEnvironment(env,log)
sensor=rltorch.BatchVectorSensor(env.observation_space)
policy=rltorch.RandomPolicy(env.observation_space,env.action_space,sensor)

local size_input=env.observation_space:size()[1]
print("Inpust size is "..size_input)
local nb_actions=env.action_space.n
print("Number of actions is "..nb_actions)

-- Creating the policy module
local module_policy=nn.Sequential():add(nn.Linear(size_input,nb_actions)):add(nn.SoftMax()):add(nn.ReinforceCategorical())
module_policy:reset(0.01)
  

MAX_LENGTH=100
DISCOUNT_FACTOR=0.9

local arguments={
    policy_module = module_policy,
    max_trajectory_size = MAX_LENGTH,
    optim=optim.sgd,
    optim_params= {
        learningRate =  0.01,
        learningRateDecay = 0,
        weightDecay = 0,
        momentum = 0.
      },
    scaling_reward=1.0
  }
  
local policy=rltorch.PolicyGradient(env.observation_space,env.action_space,sensor,arguments)


for i=1,100 do
  print("Starting episode "..i)
    policy:new_episode(env:reset())  
    local sum_reward=0.0
    local current_discount=1.0
    
    for t=1,MAX_LENGTH do  
      env:render{mode="empty"}      
      local action=policy:sample()      
      local observation,reward,done,info=unpack(env:step(action))    
      policy:feedback(reward) -- the immediate reward is provided to the policy
      policy:observe(observation)      
      sum_reward=sum_reward+current_discount*reward -- comptues the discounted sum of rewards
      current_discount=current_discount*DISCOUNT_FACTOR      
      if (done) then
        break
      end
    end
    policy:end_episode(sum_reward) -- The feedback provided for the whole episode here is the discounted sum of rewards      
end
env:close()