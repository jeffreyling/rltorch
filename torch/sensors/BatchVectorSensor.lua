 
 
 local BatchVectorSensor = torch.class('rltorch.BatchVectorSensor','rltorch.Sensor'); 

--- Transform a single (n) torch.Tensor to a (1,n) torch Tensor.
function BatchVectorSensor:__init(observation_space)
  rltorch.Sensor.__init(self,observation_space)
  self.module=nn.Reshape(1,self.observation_space:size()[1])
end

function BatchVectorSensor:process(observation)    
  return(self.module:forward(observation))
end

