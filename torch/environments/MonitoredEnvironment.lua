
local MonitoredEnvironment = torch.class('rltorch.MonitoredEnvironment','rltorch.Environment'); 
  
 
--- Initialize the environment
function MonitoredEnvironment:__init(env,log,discount_factor)
  rltorch.Environment.__init(self)
  
  if (discount_factor==nil) then self.discount_factor=1 else self.discount_factor=discount_factor end
  self.env=env
  self.action_space=env.action_space
  self.observation_space=env.observation_space
  
  self.log=log 
  self.log:addFixedParameters({environment=torch.typename(self.env)})
  self.first=true
  --self.log:newIteration()
end
 
---Update the environment given that one agent has chosen one action
-- @params agent_action the action of the agent
-- @returns observation,reward,done,info
function MonitoredEnvironment:step(agent_action)
  local r=self.env:step(agent_action)  
  self.cumul_reward=self.cumul_reward+self.current_discount_factor*r[2]
  self.current_discount_factor=self.current_discount_factor*self.discount_factor
  self.t=self.t+1
  return r
end

---- Returns the initial domain 
-- @return the action domain
function MonitoredEnvironment:reset()
   if (not self.first) then
    log:addValue("reward",self.cumul_reward)
    log:addValue("length",self.t)
   end
   self.first=false
   self.log:newIteration()

   self.t=0
   self.cumul_reward=0  
   self.current_discount_factor=1
   return self.env:reset()
end 

--- Tells if we are in a terminal state or not
function MonitoredEnvironment:close()
  log:addValue("reward",self.cumul_reward)
  log:addValue("length",self.t)
  log:newIteration()
  
  return self.env:close()
end

--- Clone the environment
function MonitoredEnvironment:render(arg)
  return self.env:render(arg)
end
