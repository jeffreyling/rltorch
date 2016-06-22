-- require 'rltorch'
 
 --- Describe an (observation/action) space
local Space = torch.class('rltorch.Space'); 
 
function Space:__init()
end
 
---- Sample one element of the space with a uniform distribution
function Space:sample()
  assert(false,"Space:sample")
end

---- Returns true of the space contains this element
function Space:contains(x)
   assert(false,"Space:contains")
end 
 
