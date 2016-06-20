require 'nn'
require 'dpnn'  
  
  --- Describe a sequential environment with one or many agents
local PolicyGradient = torch.class('rltorch.PolicyGradient','rltorch.Policy'); 

--- the policy_module is a R^n -> nb_actions >sampling vector
--- ARGUMENTS= 
----- policy_module = the policy module (takes a 1*n matrix to a 1*n vector with one for the chosen action
----- max_trajectory_size  = the size max of the trajectory (needed for copying the module)
----- optim = the optim method
----- optim_params = the optim state
----- scaling_reward = the scaling factor for the reward
----- arguments.size_memory_for_bias = number of steps to aggregate for computing the bias uin policy gradient
function PolicyGradient:__init(observation_space,action_space,sensor,arguments)
  rltorch.Policy.__init(self,observation_space,action_space) 
  self.sensor=sensor
    
  assert(arguments.policy_module~=nil)
  assert(arguments.max_trajectory_size~=nil)
  assert(arguments.optim~=nil)
  assert(arguments.optim_params~=nil)
  
  self.memory_reward=torch.Tensor(1):fill(0)
  self.memory_reward_position=1
  self.memory_reward_size=arguments.size_memory_for_bias
  
  self.optim=arguments.optim
  self.optim_params=arguments.optim_params
  
  if (arguments.scaling_reward==nil) then self.scaling_reward=1 else self.scaling_reward=arguments.scaling_reward end 
  
  self.policy_module=arguments.policy_module
  self.max_trajectory_size=arguments.max_trajectory_size
  self:init()
end

function PolicyGradient:init()    
  self.params, self.grad = rltorch.ModelsUtils():combine_all_parameters(self.policy_module) 
  print("== PolicyGradient : Cloning module "..self.max_trajectory_size.." times.")
  self.modules=rltorch.ModelsUtils():clone_many_times(self.policy_module,self.max_trajectory_size)
  print("== PolicyGradient : Cloning DONE")
  self.delta=torch.Tensor(1,self.action_space.n):fill(1)
  
  self.feval = function(params_new)
    if self.params ~= params_new then
        self.params:copy(params_new)
    end
    
    self.grad:zero()
    if (self.memory_reward_position>self.memory_reward:size(1)) then self.memory_reward:resize(self.memory_reward_position) end
    self.memory_reward[self.memory_reward_position]=self.reward_trajectory
    self.memory_reward_position=self.memory_reward_position+1
    if (self.memory_reward_position>self.memory_reward_size) then self.memory_reward_position=1 end    
    local avg_reward=self.memory_reward:mean()
    print("AVG Reward = "..avg_reward.." pour "..self.memory_reward:size(1))
    sum_reward=torch.Tensor(1):fill(self.reward_trajectory-avg_reward)
    
    for t=1,self.trajectory:get_number_of_observations()-1 do
      local out=self.modules[t].output
      self.modules[t]:reinforce(sum_reward)
      self.modules[t]:backward(self.trajectory.observations[t],self.delta)
    end
             
    return -sum_reward,self.grad           
  end
end

function PolicyGradient:new_episode(initial_observation,informations)
  self.trajectory=rltorch.Trajectory()
  self.last_sensor=self.sensor:process(initial_observation):clone()
  self.trajectory:push_observation(self.last_sensor)
end

function  PolicyGradient:observe(observation)  
  self.last_sensor=self.sensor:process(observation):clone()
  --print(self.last_sensor)
  self.trajectory:push_observation(self.last_sensor)  
end

function PolicyGradient:feedback(reward)
  self.trajectory:push_feedback(reward)
end

function PolicyGradient:sample()
  local out=self.modules[self.trajectory:get_number_of_observations()]:forward(self.last_sensor)
  local vmax,imax=out:max(2)
  return(imax[1][1])
end

function PolicyGradient:end_episode(feedback)
  ---- Launch the gradient optimization method
  self.reward_trajectory=feedback*self.scaling_reward  
  local _,fs=self.optim(self.feval,self.params,self.optim_params) 
end

 
