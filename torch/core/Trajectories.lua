tds=require 'tds'  
   
   
--- Describe a memory of trajectories

local Trajectories = torch.class('rltorch.Trajectories'); 

function Trajectories:__init(observation_space,action_space)
  self.observation_space=observation_space
  self.action_space=action_space
  self.trajectories=tds.Vec()
end

function Trajectories:new_trajectory()
  self.trajectories[#self.trajectories+1]=rltorch.Trajectory()
  self.ct=self.trajectories[#self.trajectories-1]
end
  
function Trajectories:push_observation(o)
  self.cf.push_observation(o)
end

  
function Trajectories:push_action(o)
  self.cf.push_action(o)
end
  
function Trajectories:push_feedback(o)
  self.cf.push_feedback(o)
end
  
function Trajectories:push_done(o)
  self.cf.push_done(o)
end

function Trajectories:get_number_of_observations()
  return( #self.observations)
end
  
function Trajectories:size()
  return(#self.trajectories)
end

function Trajectories:clear()
  self.trajectories=tds.Vec()
end
 
