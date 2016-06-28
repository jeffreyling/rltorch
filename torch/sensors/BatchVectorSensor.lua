
 
local BatchVectorSensor = torch.class('rltorch.BatchVectorSensor','rltorch.Sensor'); 

--- Transform a single (n) torch.Tensor to a (1,n) torch Tensor.
function BatchVectorSensor:__init(observation_space)
  rltorch.Sensor.__init(self,observation_space)
  
  if (torch.type(observation_space)=="rltorch.Box") then
    self.current_size=observation_space.low:nElement()
    self.module=nn.Reshape(1,self.current_size,false)
  else
    self.current_size=nil
  end
end

function BatchVectorSensor:process(observation)
  local n=observation:nElement()
  if (self.current_size~=n) then self.module=nn.Reshape(1,n,false); self.current_size=n end
  
  local out=self.module:forward(observation)
  return out
end

function BatchVectorSensor:size()
  return(self.current_size)
end
