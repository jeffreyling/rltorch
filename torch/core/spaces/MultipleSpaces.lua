  
 --- Describe a multiple spaces domain (basically: no information, and the domain of the observation changes)
local MultipleSpaces = torch.class('rltorch.MultipleSpaces','rltorch.Space'); 
 
function MultipleSpaces:__init()
  
end
 
function MultipleSpaces:sample()
  assert(false,"Impossible to sample in rltorch.MultipleSpaces")
end

function MultipleSpaces:contains(x)
  assert(false,"Impossible to use 'contains' in rltorch.MultipleSpaces")
end 

