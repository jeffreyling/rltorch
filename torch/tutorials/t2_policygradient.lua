

require ('optim') 
require('rltorch')

MAX_LENGTH=1000
DISCOUNT_FACTOR=1.0
NB_TRAJECTORIES=10000

--env = rltorch.MountainCar_v0()
env = rltorch.CartPole_v0()
math.randomseed(os.time())
--env = rltorch.EmptyMaze_v0(10,10)
sensor=rltorch.BatchVectorSensor(env.observation_space)
--sensor=rltorch.TilingSensor2D(env.observation_space,30,30)

local size_input=sensor:size()
local nb_actions=env.action_space.n

-- Creating the policy module
local module_policy=nn.Sequential():add(nn.Linear(size_input,nb_actions)):add(nn.SoftMax()):add(nn.ReinforceCategorical())
--local module_policy=nn.Sequential():add(nn.Linear(size_input,nb_actions*2)):add(nn.Tanh()):add(nn.Linear(nb_actions*2,nb_actions)):add(nn.SoftMax()):add(nn.ReinforceCategorical())
module_policy:reset(0.01)



local arguments={
    policy_module = module_policy,
    max_trajectory_size = MAX_LENGTH,
    optim=optim.adam,
    optim_params= {
        learningRate =  0.01  
      },
    scaling_reward=1.0/MAX_LENGTH,
    size_memory_for_bias=100
  }
  
policy=rltorch.PolicyGradient(env.observation_space,env.action_space,sensor,arguments)

local rewards={}
for i=1,NB_TRAJECTORIES do
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
    
    rewards[i]=sum_reward
    print("Reward at "..i.." is "..sum_reward)
    if (i%100==0) then gnuplot.plot(torch.Tensor(rewards),"|") end
    
    policy:end_episode(sum_reward) -- The feedback provided for the whole episode here is the discounted sum of rewards      
end
env:close()