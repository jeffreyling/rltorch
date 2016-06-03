 
require('rltorch')
log=rltorch.ExperimentLogConsole()
log=rltorch.ExperimentLogCSV(false,"tmp/","now")

env = rltorch.MountainCar_v0()
env=rltorch.MonitoredEnvironment(env,log)
sensor=rltorch.IdSensor()
policy=rltorch.RandomPolicy(env.observation_space,env.action_space,sensor)

MAX_LENGTH=100
DISCOUNT_FACTOR=0.9

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