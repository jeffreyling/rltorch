 require 'logroll'
 require 'json'
 
local ExperimentLogConsole = torch.class('rltorch.ExperimentLogConsole','rltorch.ExperimentLog'); 

function ExperimentLogConsole:__init(memory)
  rltorch.ExperimentLog.__init(self,memory)
  self.log=logroll.print_logger()
end

function ExperimentLogConsole:newIteration()
  if (self.iteration==0) then io.write(json.encode(self.parameters)); io.write("\n") end
  if (not rltorch.ExperimentLog.isEmpty(self)) then
    io.write("Iteration "..self.iteration.."\n")
    io.write(json.encode(self.currentjson)); io.write("\n")
  end
  rltorch.ExperimentLog.newIteration(self)
end


function ExperimentLogConsole:addDescription(text)  
  io.stdout:write(text)
end
