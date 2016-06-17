require 'image'
 
 local BatchVectorSensor_ForAtari = torch.class('rltorch.BatchVectorSensor_ForAtari','rltorch.Sensor'); 



--- Transform a single (n) torch.Tensor to a (1,n) torch Tensor.
function BatchVectorSensor_ForAtari:__init(observation_space,height,width,flag_grayscale)
  rltorch.Sensor.__init(self,observation_space)
  
  assert(self.observation_space:size()[1]==3)
  self.width=width
  self.height=height
  
  if (flag_grayscale) then 
    self.gs=true 
    self._size=width*height
    self.bandw=torch.Tensor(self.width,self.height)
  else
    self._size=self.observation_space:size()[1]*width*height
  end
  
  self.module=nn.Reshape(1,self._size,false)
  self.rescaled=torch.ByteTensor(3,self.width,self.height)
  self.rescaled_double=torch.Tensor(3,self.width,self.height) 
end

function BatchVectorSensor_ForAtari:process(observation)
 
  -- Rescale
  local resc=self.rescaled
  if ((self.height~=observation:size(2)) or (self.width~=observation:size(3))) then
     image.scale(resc,observation,"simple")
  else 
    resc=observation
  end
  
  -- Conversion to double tensor
  self.rescaled_double:copy(resc):div(255)
  
  --- Gray ? 
  if (self.gs) then
    self:rgb2gray(self.rescaled_double)
    return self.module:forward(self.bandw)
  else  
    return self.module:forward(self.rescaled_double)
  end
end

function BatchVectorSensor_ForAtari:size()
  return(self._size)
end

-- convert rgb to grayscale by averaging channel intensities
function BatchVectorSensor_ForAtari:rgb2gray(im)
	local dim, w, h = im:size()[1], im:size()[2], im:size()[3]
	assert(dim == 3,'<error> expected 3 channels')

	-- a cool application of tensor:select
	local r = im:select(1, 1)
	local g = im:select(1, 2)
	local b = im:select(1, 3)

	local z = self.bandw
  z:copy(r):mul(0.21)
	z:add(0.72, g)
	z:add(0.07, b)
end


