local EmptyMaze_v0 = torch.class('rltorch.EmptyMaze_v0','rltorch.Environment'); 
  
 
--- Empty maze with an initial position, and a goal

function EmptyMaze_v0:__init(size_x,size_y,start_x,start_y,goal_x,goal_y)
  self.size_x=size_x
  self.size_y=size_y
  
  if (start_x==nil) then
      self.start_x=math.random(self.size_x)
      self.start_y=math.random(self.size_y)
      self.goal_x=math.random(self.size_x)
      self.goal_y=math.random(self.size_y)
      while ((self.goal_x==self.start_x) and (self.goal_y==self.start_y)) do
        self.goal_x=math.random(self.size_x)
        self.goal_y=math.random(self.size_y)
      end
  else
    self.start_x=start_x
    self.start_y=start_y
    self.goal_x=goal_x
    self.goal_y=goal_y
  end
  print("EmptyMaze_v0: starting position = "..self.start_x..";"..self.start_y.." Goal = "..self.goal_x..";"..self.goal_y)
  
  local vlow=torch.Tensor({1,1})
  local vhigh=torch.Tensor({self.size_x,self.size_y})
  self.action_space = rltorch.Discrete(4)
  self.observation_space = rltorch.Box(vlow,vhigh)
end
 
function EmptyMaze_v0:step(agent_action)  
  local x=self.state[1]
  local y=self.state[2]
  if (agent_action==1) then
    x=x-1
  elseif (agent_action==2) then
    x=x+1
  elseif (agent_action==3) then
    y=y-1
  elseif (agent_action==4) then
    y=y+1
  else
    assert(false,"action is not a valid one")
  end
  
  if (x<1) then x=1 end
  if (x>self.size_x) then x=self.size_x end
  if (y<1) then y=1 end
  if (y>self.size_y) then y=self.size_y end
  
  local reward=0
  local done=false
  if ((x==self.goal_x) and (y==self.goal_y)) then reward=1; done=true end
  
  self.state[1]=x
  self.state[2]=y
  return {self.state,reward,done}
end

function EmptyMaze_v0:reset()
   self.state = torch.Tensor({self.start_x,self.start_y})
   return(self.state)
end 

function EmptyMaze_v0:close()  
end

function EmptyMaze_v0:render(arg)
  if (arg.mode=="console") then
    print("Position = "..self.state[1].." ; "..self.state[2])
  end
end
