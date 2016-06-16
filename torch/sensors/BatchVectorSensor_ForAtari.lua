require 'image'
 
 local BatchVectorSensor_ForAtari = torch.class('rltorch.BatchVectorSensor_ForAtari','rltorch.Sensor'); 

--- Transform a single (n) torch.Tensor to a (1,n) torch Tensor.
function BatchVectorSensor_ForAtari:__init(observation_space,height,width)
  rltorch.Sensor.__init(self,observation_space)
  
  local ss=self.observation_space:size():size()
   self._size=self.observation_space:size()[1]*width*height 
   self.width=width
   self.height=height
  self.module=nn.Reshape(1,self._size,false)
  self.retour=torch.ByteTensor(3,self.width,self.height)
  self.tmp=torch.Tensor(3,self.width,self.height)    
end

function BatchVectorSensor_ForAtari:process(observation)
  --self.tmp:copy(observation):div(255)
  
  if ((self.height~=observation:size(2)) or (self.width~=observation:size(3))) then
     image.scale(self.retour,observation,"simple")
  else 
    self.retour=observation
  end
  self.tmp:copy(self.retour):div(255)
  return self.module:forward(self.tmp)
end

function BatchVectorSensor_ForAtari:size()
  return(self._size)
end
