--- This class implements a set of tools for (easy) RL experiments

local RLTools = torch.class('rltorch.RLTools'); 
   
--- This function launches nb_trajectories trajectories and save the total discounted reward + trajectories lengths in a log (see tutorials/t2_policygradient_bis.lua)
----- policy : the policy
----- environment : the environment
----- lig : the log object

--- parameters:
----- discount_factor : the discount_factor (deault 1)
----- nb_trajectories : the number of trajectories (default 1000)
----- size_max_trajectories : the maximum size of each trajectory (default nil => + infinity)
----- render_parameters: if one wants to render the environment, then you have to provide the render parameters
----- display_every : display the reward evry n iterations

--- The policy receive the immedaite reward, and the discounted sum of reward at the end of the episode
function RLTools:monitorReward(environment,policy,log,parameters)
  if (parameters.discount_factor==nil) then parameters.discount_factor=1 end
  if (parameters.nb_trajectories==nil) then parameters.nb_trajectories=1000 end
  print(policy)
  local rewards={}
  for i=1,parameters.nb_trajectories do
    local out=env:reset()
    policy:new_episode(out)  
    local sum_reward=0.0
    local current_discount=1.0
    local flag=true
    local current_iteration=0
    while(flag) do  
      if (parameters.render_parameters~=nil) then env:render(parameters.render_parameters) end
      
      local action=policy:sample()      
      local observation,reward,done,info=unpack(env:step(action))    
      policy:feedback(reward) -- the immediate reward is provided to the policy
      policy:observe(observation)    
      sum_reward=sum_reward+current_discount*reward -- comptues the discounted sum of rewards
      current_discount=current_discount*parameters.discount_factor      
      current_iteration=current_iteration+1
      if (done) then
        flag=false
      end
      if (parameters.size_max_trajectories~=nil) then
        if (current_iteration>=parameters.size_max_trajectories) then 
          flag=false end
      end
    end
    policy:end_episode(sum_reward) -- The feedback provided for the whole episode here is the discounted sum of rewards      
    
    log:newIteration()
    log:addValue("length",current_iteration)
    log:addValue("reward",sum_reward)
    
    rewards[i]=sum_reward
    if (parameters.display_every~=nil) then
      if (i%parameters.display_every==0) then gnuplot.plot(torch.Tensor(rewards),"|") end
    end    
  end
  env:close()
  log:newIteration()
end
