 
 --- The gaol of a sensor is to transform an observation to a data structure that can be processed by a policy
 local Sensor = torch.class('rltorch.Sensor'); 

function Sensor:__init(observation_space)
  self.observation_space=observation_space
end

function Sensor:process(observation)
  assert(false,"No process function")
end

