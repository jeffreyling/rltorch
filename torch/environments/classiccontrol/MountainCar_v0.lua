
local MountainCar_v0 = torch.class('rltorch.MountainCar_v0','rltorch.Environment'); 
  
 
--- Initialize the environment
function MountainCar_v0:__init()
  self.min_position = -1.2
  self.max_position = 0.6
  self.max_speed = 0.07
  self.goal_position = 0.5

  self.low = torch.Tensor({self.min_position, -self.max_speed})
  self.high = torch.Tensor({self.max_position, self.max_speed})
  self.viewer = None

  self.action_space = rltorch.Discrete(3)
  self.observation_space = rltorch.Box(self.low, self.high)
end
 
---Update the environment given that one agent has chosen one action
-- @params agent_action the action of the agent
-- @returns observation,reward,done,info
--- Here action is 1 2 or 3
function MountainCar_v0:step(agent_action)  
  local position=self.state[1]
  local velocity=self.state[2]
  velocity = velocity+ (agent_action-2)*0.001 + math.cos(3*position)*(-0.0025)
  if (velocity > self.max_speed) then velocity = self.max_speed end
  if (velocity < -self.max_speed)then velocity = -self.max_speed end
  position = position + velocity
  if (position > self.max_position)then position = self.max_position end
  if (position < self.min_position)then position = self.min_position end
  if (position==self.min_position and velocity<0) then velocity = 0 end

  local done = (position >= self.goal_position)
  local reward = -1.0

  self.state[1] = position
  self.state[2] = velocity
  return {self.state,reward,done}
end

---- Returns the initial domain 
-- @return the action domain
function MountainCar_v0:reset()
   self.state = torch.Tensor({math.random()*(-0.4+0.6)-0.4, 0})
   return(self.state)
end 

--- Tells if we are in a terminal state or not
function MountainCar_v0:close()
  
end

--- Clone the environment
function MountainCar_v0:render(arg)
  if (arg.mode=="console") then
    print("Position = "..self.state[1].." / Speed = "..self.state[2])
  end
end
