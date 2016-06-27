 
 
local Policy = torch.class('rltorch.Policy'); 

--- Describe a policy

--- Initialize the policy given the observation and action spaces
-- See Space.lua
function Policy:__init(observation_space,action_space,sensor)
  self.observation_space=observation_space
  self.action_space=action_space
  self.sensor=sensor
  self.train=true -- Default is training mode
end

--- This function must be called at the begining of a new episode
function Policy:new_episode(initial_observation,informations)
end

--- This function must be called before sample
function  Policy:observe(observation)  
end

--- This function corresponds to the immediate feedback (e.g reward) received after each action
function Policy:feedback(reward)
end

--- Sample an action 
function Policy:sample()
  assert(false,"No sample function implemented for this policy")
end

--- Finishes an episode. One can provide a feedback corresponding to the whole episode (e.g total reward)
function Policy:end_episode(feedback)
end

