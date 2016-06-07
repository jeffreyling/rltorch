 
 
 local IdSensor = torch.class('rltorch.IdSensor','rltorch.Sensor'); 

function IdSensor:__init(observation_space)
  rltorch.Sensor.__init(self,observation_space)
end

function IdSensor:process(observation)  
  return(observation)
end

