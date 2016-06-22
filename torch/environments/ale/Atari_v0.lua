require 'torch'

local Atari_v0 = torch.class('rltorch.Atari_v0','rltorch.Environment'); 


function Atari_v0:__init(parameters)
  rltorch.Environment.__init(self,parameters)
  
  if (self.parameters.height==nil) then self.parameters.height=210 end
  if (self.parameters.width==nil) then self.parameters.width=160 end
  self.RAM_LENGTH=128
  
  self.config = {
        gameOverReward=0,
        enableRamObs=false,
    }
  self.win = nil
  self.ale = alewrap.newAle(self.parameters.rom)
  local obsShapes = {{self.parameters.height, self.parameters.width}}
  if self.config.enableRamObs then
      obsShapes={{self.parameters.height, self.parameters.width}, {self.RAM_LENGTH}}
  end
  
  self.action_space = rltorch.Discrete(18)
  self.observation_space = rltorch.Box(0,255,torch.LongStorage({3,self.parameters.height,self.parameters.width}))
  self.obsShapes=obsShapes
    
    
  self.obs = torch.ByteTensor(self.parameters.height, self.parameters.width)    
  
  self.ale:resetGame() 
end

 
function Atari_v0:reset()
  local v=self.ale:resetGame()
  self.ale:fillObs(torch.data(self.obs), self.obs:nElement())    
  self.last_observation=alewrap.getRgbFromPalette(self.obs)  
  return self.last_observation
end  


function Atari_v0:step(agent_action)
  local reward = self.ale:act(agent_action-1)
  local done=self.ale:isGameOver()
  self.ale:fillObs(torch.data(self.obs), self.obs:nElement())    
  self.last_observation=alewrap.getRgbFromPalette(self.obs)  
  
  return {self.last_observation,reward,done}
end



function Atari_v0:close()
  
end

function Atari_v0:render(arg)
  if (arg.mode=="console") then
    print("Cannot render Atari in console")
  elseif (arg.mode=="qt") then
    self.win=image.display({image=self.last_observation,win=self.win})
  end
end

--ACTION_MEANING = {
--    0 : "NOOP",
--    1 : "FIRE",
--    2 : "UP",
--    3 : "RIGHT",
--    4 : "LEFT",
--    5 : "DOWN",
--    6 : "UPRIGHT",
--    7 : "UPLEFT",
--    8 : "DOWNRIGHT",
--    9 : "DOWNLEFT",
--    10 : "UPFIRE",
--    11 : "RIGHTFIRE",
--    12 : "LEFTFIRE",
--    13 : "DOWNFIRE",
--    14 : "UPRIGHTFIRE",
--    15 : "UPLEFTFIRE",
--    16 : "DOWNRIGHTFIRE",
--    17 : "DOWNLEFTFIRE",
--}
