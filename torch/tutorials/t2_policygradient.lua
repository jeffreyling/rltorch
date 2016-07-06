

require ('optim') 
require('rltorch')

MAX_LENGTH=1000
DISCOUNT_FACTOR=1.0
NB_TRAJECTORIES=100
batch_size = 10

-- Want to do batch mode but seems troublesome with this env setup...
-- We can just do multiple pa

--env = rltorch.MountainCar_v0()
env = rltorch.CartPole_v0()
math.randomseed(3435)
--env = rltorch.EmptyMaze_v0(10,10)
sensor=rltorch.BatchVectorSensor(env.observation_space, batch_size)
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
    size_memory_for_bias=100
  }
  
policy=rltorch.PolicyGradient(env.observation_space,env.action_space,sensor,arguments)

local rewards={}
for i=1,NB_TRAJECTORIES do
  print("Starting episode "..i)
    policy:new_episode(env:reset())  
    local sum_reward= torch.zeros(batch_size)
    local current_discount=1.0
    
    for t=1,MAX_LENGTH do  
      env:render{mode="empty"}      
      local action=policy:sample()      
      local observations,rewards,dones,info=unpack(env:step(action))    
      policy:feedback(rewards) -- the immediate reward is provided to the policy
      policy:observe(observations)      
      sum_reward:add(current_discount, reward) -- comptues the discounted sum of rewards
      current_discount=current_discount*DISCOUNT_FACTOR      
      local all_done = true
      for j=1,batch_size do
        if dones[j] == 0 then
          all_done = false
          break
        end
      end

      if (all_done) then        
        break
      end
    end
    
    rewards[i]=sum_reward
    print("Reward at "..i.." is "..sum_reward)
    --if (i%100==0) then gnuplot.plot(torch.Tensor(rewards),"|") end
    
    policy:end_episode(sum_reward) -- The feedback provided for the whole episode here is the discounted sum of rewards      
end
env:close()
