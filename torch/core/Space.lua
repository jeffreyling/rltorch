-- require 'rltorch'
 
 --- Describe a sequential environment with one or many agents
local Space = torch.class('rltorch.Space'); 
 
--- Initialize the environment
function Space:__init()
end
 
---Update the environment given that one agent has chosen one action
-- @params agent_action the action of the agent
-- @returns observation,reward,done,info
function Space:sample()
  assert(false,"Space:sample")
end

---- Returns the initial domain 
-- @return the action domain
function Space:contains(x)
   assert(false,"Space:contains")
end 
 
