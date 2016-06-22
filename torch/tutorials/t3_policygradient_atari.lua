require ('optim') 
require('rltorch')
package.path = '/home/denoyer/alewrap/?/init.lua;' .. package.path
local alewrap = require 'alewrap'


MAX_LENGTH=300  
DISCOUNT_FACTOR=1.0
NB_TRAJECTORIES=10000

--env = rltorch.MountainCar_v0()
env = rltorch.Atari_Breakout_v0({rom="environments/ale/roms/breakout.bin"})
math.randomseed(os.time())
sensor=rltorch.BatchVectorSensor_ForAtari(env.observation_space,40,30,true)

local size_input=sensor:size()
print("Inpust size is "..size_input)
local nb_actions=env.action_space.n
print("Number of actions is "..nb_actions)

-- Creating the policy module
--  module_policy=nn.Sequential():add(nn.Linear(size_input,size_input/2)):add(nn.Tanh()):add(nn.Linear(size_input/2,nb_actions)):add(nn.SoftMax()):add(nn.ReinforceCategorical())
module_policy=nn.Sequential():add(nn.Linear(size_input,nb_actions)):add(nn.SoftMax()):add(nn.ReinforceCategorical())
module_policy:reset(0.001)

local arguments={
    policy_module = module_policy,
    max_trajectory_size = MAX_LENGTH,
    optim=optim.adam,
    optim_params= {
        learningRate =  0.001  
      },
    scaling_reward=1.0, --/MAX_LENGTH,
    size_memory_for_bias=100
  }
  
--policy=rltorch.RandomPolicy(env.observation_space,env.action_space,sensor)
policy=rltorch.PolicyGradient(env.observation_space,env.action_space,sensor,arguments)

local rewards={}


for i=1,NB_TRAJECTORIES do
  print("Starting episode "..i)
    policy:new_episode(env:reset())  
    local sum_reward=0.0
    local current_discount=1.0
    
    for t=1,MAX_LENGTH do  
     if (i%100==1) then env:render{mode="qt",fps=nil}       end
     -- env:render{mode="human"}      
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
    
    rewards[i]=sum_reward
    if (i%1==0) then gnuplot.plot(torch.Tensor(rewards),"|") end
    
    policy:end_episode(sum_reward) -- The feedback provided for the whole episode here is the discounted sum of rewards      
end
env:close()