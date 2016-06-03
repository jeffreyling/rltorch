 
 
 local Sensor = torch.class('rltorch.Sensor'); 

function Sensor:__init(observation_space)
  self.observation_space=observation_space
end

function Sensor:process(observation)
  assert(false,"No process function")
end

