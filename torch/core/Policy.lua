 
 
  --- Describe a sequential environment with one or many agents
local Policy = torch.class('rltorch.Policy'); 


function Policy:__init(observation_space,action_space,sensor)
  self.observation_space=observation_space
  self.action_space=action_space
  self.sensor=sensor
end

function Policy:new_episode(initial_observation,informations)
end

function  Policy:observe(observation)  
end

function Policy:feedback(reward)
end

function Policy:sample()
  assert(false,"No sample function implemented for this policy")
end

function Policy:end_episode(feedback)
end

