 
 --- Describe a sequential environment with one or many agents
local Discrete = torch.class('rltorch.Discrete','rltorch.Space'); 
 
--- Initialize the environment
function Discrete:__init(n)
  self.n=n
  
end
 
---Update the environment given that one agent has chosen one action
-- @params agent_action the action of the agent
-- @returns observation,reward,done,info
function Discrete:sample()
  return(math.random(self.n))
end

---- Returns the initial domain 
-- @return the action domain
function Discrete:contains(x)
   assert(x==math.floor(x))
   if ((x>=1) and (x<=self.n)) then return true else return false end
end 


