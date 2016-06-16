
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
   self.state = torch.Tensor({math.random()*(-0.4+0.6)-0.4, math.random()*(2*self.max_speed)-self.max_speed})
   return(self.state)
end 

--- Tells if we are in a terminal state or not
function MountainCar_v0:close()
  
end


function MountainCar_v0:_height(xs)
  return math.sin(3 * xs)*0.45
end

--- Clone the environment
function MountainCar_v0:render(arg)
  if (arg.mode=="console") then
    print("Position = "..self.state[1].." / Speed = "..self.state[2])
    elseif (arg.mode=="qt") then
      local SX=320
      local SY=240
      local POLE_SIZE=SY/5
      if (self.__render_widget==nil) then 
        require 'qt'
        require 'qtuiloader'
        require 'qtwidget'
        
        self.__render_widget = qtwidget.newwindow(SX,SY,"MountainCar_v0")
        self.__render_widget:setangleunit("Degrees")
      end
      
    ---
    local scale_y=SY*0.5
    self.__render_widget:showpage()
    self.__render_widget:setcolor("black")
    do
      local pos=self.min_position
      local py=SY/2*scale_y+self:_height(pos)
      self.__render_widget:fill(false)  
      self.__render_widget:moveto(1,py)
      self.__render_widget:stroke()
      for px=2,SX,10 do
          pos=px/SX*(self.max_position-self.min_position)+self.min_position
          py=SY/2-scale_y*self:_height(pos)
          self.__render_widget:lineto(px,py)
      end
      self.__render_widget:stroke()      
      
      self.__render_widget:setcolor("red")
      local ppx=self.state[1]
      local pos=((ppx-self.min_position)/(self.max_position-self.min_position))*SX
      local ppy=SY/2-scale_y*self:_height(ppx)
      
      self.__render_widget:arc(math.floor(pos)+1,math.floor(ppy),SX/100.0,0,360)
      self.__render_widget:fill(true)  
      self.__render_widget:stroke()
      
    end
    sys.sleep(1.0/arg.fps)
    self.__render_widget:painter()
    
  end
end
