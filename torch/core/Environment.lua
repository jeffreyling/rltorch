-- require 'rltorch'
 
 --- Describe a sequential environment with one or many agents
local Environment = torch.class('rltorch.Environment'); 
 
--- Initialize the environment
function Environment:__init(parameters)
  self.parameters=parameters
end
 
---Update the environment given that one agent has chosen one action
-- @params agent_action the action of the agent
-- @returns observation,reward,done,info
function Environment:step(agent_action)
  assert(false,"Environment:__init")
end

---- Returns the initial domain 
-- @return the action domain
function Environment:reset()
   assert(false,"Environment:reset")
end 

--- Tells if we are in a terminal state or not
function Environment:close()
  
end

--- Clone the environment
function Environment:render(...)
  assert(false,"Environment:render")
end
