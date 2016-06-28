 
require 'nn'
require 'dpnn'  
  
  --- A policy for the MulticlassClassification Environment which emulates a SGD algorithm. The SGD is made at the end of each episode. It can also be used for imitation learning. It is based on a NLL Criterion. At each timestep, the feedback method must be used with a true_category value (not a reward) i.e the action that the policy would have taken....
local StochasticGradientImitationPolicy = torch.class('rltorch.StochasticGradientImitationPolicy','rltorch.Policy'); 

--- the policy_module is a R^n -> nb_actions >sampling vector
--- ARGUMENTS= 
----- policy_module = the policy module (takes a 1*n matrix to a 1*n vector of scores)
----- optim = the optim method (e.g optim.adam)
----- optim_params = the optim initial state 
function StochasticGradientImitationPolicy:__init(observation_space,action_space,sensor,arguments)
  rltorch.Policy.__init(self,observation_space,action_space) 
  self.sensor=sensor
    
  assert(arguments.policy_module~=nil)
  assert(arguments.optim~=nil)
  assert(arguments.optim_params~=nil)
  self.arguments=arguments
  
  self:init()
end

function StochasticGradientImitationPolicy:init()    
  self.params, self.grad = rltorch.ModelsUtils():combine_all_parameters(self.arguments.policy_module) 
  self.criterion=nn.CrossEntropyCriterion()
  self.sm=nn.Sequential():add(nn.SoftMax()):add(nn.ReinforceCategorical())
  self.feval = function(params_new)
    if self.params ~= params_new then
       self.params:copy(params_new)
    end
    
    self.grad:zero()
    
    --First, build the minibatch based on the trajectory
    local input=torch.Tensor(self.trajectory:get_number_of_observations()-1,self.trajectory.observations[1]:size(2))
    local ground_truth=torch.Tensor(self.trajectory:get_number_of_observations()-1,1)
    for i=1,self.trajectory:get_number_of_observations()-1 do
      input[i]:copy(self.trajectory.observations[i][1])
      ground_truth[i][1]=self.trajectory.feedback[i]
    end
    
    local out=self.arguments.policy_module:forward(input)
    local loss=self.criterion:forward(out,ground_truth)
    local delta=self.criterion:backward(out,ground_truth)    
    self.arguments.policy_module:backward(input,delta)
                 
    return loss,self.grad           
  end
end

function StochasticGradientImitationPolicy:new_episode(initial_observation,informations)
  self.trajectory=rltorch.Trajectory()
  self.last_sensor=self.sensor:process(initial_observation):clone()
  self.trajectory:push_observation(self.last_sensor)
end

function  StochasticGradientImitationPolicy:observe(observation)  
  self.last_sensor=self.sensor:process(observation):clone()
  self.trajectory:push_observation(self.last_sensor)  
end

function StochasticGradientImitationPolicy:feedback(true_action)
  self.trajectory:push_feedback(true_action)
end

function StochasticGradientImitationPolicy:sample()
  local out=self.arguments.policy_module:forward(self.last_sensor)
  local vout=self.sm:forward(out)  
  local vmax,imax=vout:max(2)
  return(imax[1][1])
end

function StochasticGradientImitationPolicy:end_episode()
  if (self.train) then   local _,fs=self.arguments.optim(self.feval,self.params,self.arguments.optim_params)  end
end

 
