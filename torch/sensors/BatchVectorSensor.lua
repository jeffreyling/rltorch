
 
 local BatchVectorSensor = torch.class('rltorch.BatchVectorSensor','rltorch.Sensor'); 

--- Transform a single (n) torch.Tensor to a (1,n) torch Tensor.
function BatchVectorSensor:__init(observation_space)
  rltorch.Sensor.__init(self,observation_space)
  local fs=1
  local ss=self.observation_space:size():size()
  for i=1,ss do fs=fs*self.observation_space:size()[i] end  
  self._size=fs
  self.module=nn.Reshape(1,self._size,false)
end

function BatchVectorSensor:process(observation)
  local out=self.module:forward(observation)
  return out
end

function BatchVectorSensor:size()
  return(self._size)
end
