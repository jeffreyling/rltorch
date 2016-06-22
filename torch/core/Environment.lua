-- require 'rltorch'
 
 --- Describe a sequential environment with one or many agents
local Environment = torch.class('rltorch.Environment'); 
 
--- Initialize the environment (with parameters if needed)
function Environment:__init(parameters)
  self.parameters=parameters
end
 
---Update the environment given the chosen action
-- @params agent_action the action of the agent
-- @returns observation,reward,done,info
function Environment:step(agent_action)
  assert(false,"Environment:__init")
end

-- Reset the environment
function Environment:reset()
   assert(false,"Environment:reset")
end 

--- Close the environment (at the end of the process)
function Environment:close()
  
end

--- Rendering of the current environment state (depends on the environment)
function Environment:render(...)
  assert(false,"Environment:render")
end
