require 'nn'
require 'dpnn'  
  
  --- A policy based on REINFORCE for discrete action spaces

local RecurrentPolicyGradient = torch.class('rltorch.RecurrentPolicyGradient','rltorch.Policy'); 

--- the policy_module is a R^n -> nb_actions >sampling vector
--- ARGUMENTS= 
----- policy_module = the policy module (takes a 1*N matrix to a 1*A vector using the dpnn package. This module goes from the latent space to the action space)
----- recurrent_modules = the recurrent modules takes [z_t,o_t] and returns z_{t+1} (typically a GRU for example) -- you have to privude one module for each possible action
----- initial_recurrent_module = the first module that takes [initial_state, initial_boservation] and returns the first latent representation z_1
----- initial_state = the initial vector (size (1,N)) 
----- N = the dimensionnality of tha latent space
----- max_trajectory_size  = the maximum length of the trajectories
----- optim = the optim method (e.g optim.adam)
----- optim_params = the optim initial state 
----- scaling_reward = the scaling factor for the reward  
----- arguments.size_memory_for_bias = number of steps to aggregate for computing the bias in policy gradient -- the n last reward values are used to correct the reward obtained.
function RecurrentPolicyGradient:__init(observation_space,action_space,sensor,arguments)
  rltorch.Policy.__init(self,observation_space,action_space) 
  self.sensor=sensor
    
  assert(arguments.policy_module~=nil)
  assert(arguments.recurrent_modules~=nil)
  assert(arguments.initial_recurrent_module~=nil)
  assert(#arguments.recurrent_modules==action_space.n)
  assert(arguments.N~=nil)
  assert(arguments.max_trajectory_size~=nil)
  assert(arguments.optim~=nil)
  assert(arguments.optim_params~=nil)
  self.arguments=arguments
  
  self.memory_reward=torch.Tensor(1):fill(0)
  self.memory_reward_position=1
  self.memory_reward_size=arguments.size_memory_for_bias
  
  self.optim=arguments.optim
  self.optim_params=arguments.optim_params
  
  if (arguments.scaling_reward==nil) then self.scaling_reward=1 else self.scaling_reward=arguments.scaling_reward end 
  
  self.max_trajectory_size=arguments.max_trajectory_size
  self:init()
end

function RecurrentPolicyGradient:init()    
  self.params, self.grad = rltorch.ModelsUtils():combine_all_parameters(self.arguments.policy_module,self.arguments.initial_recurrent_module,unpack(self.arguments.recurrent_modules)) 
  
  -- Cloning the modules
  self.dmodules=rltorch.ModelsUtils():clone_many_times(self.arguments.policy_module,self.max_trajectory_size)
  self.rmodules={}
  for a=1,self.action_space.n do self.rmodules[a]=rltorch.ModelsUtils():clone_many_times(self.arguments.recurrent_modules[a],self.max_trajectory_size) end
  
  -- Evaluation of the gradient 
  self.feval = function(params_new)
    if self.params ~= params_new then
        self.params:copy(params_new)
    end
    
    self.grad:zero()
    
    --Store the last reward in the reward memory (for variance stuff)
    if (self.memory_reward_position>self.memory_reward:size(1)) then self.memory_reward:resize(self.memory_reward_position) end
    self.memory_reward[self.memory_reward_position]=self.reward_trajectory
    self.memory_reward_position=self.memory_reward_position+1
    if (self.memory_reward_position>self.memory_reward_size) then self.memory_reward_position=1 end    
    local avg_reward=self.memory_reward:mean()
    local sum_reward=torch.Tensor(1):fill(self.reward_trajectory-avg_reward)
    print("AVG REWARD = "..avg_reward)
    
    local delta=torch.Tensor(1,self.arguments.N):fill(0)
    -- now, we do the backward procedure given the currently stored trajectory
    for t=self.trajectory:get_number_of_observations()-1,2,-1 do      
      local chosen_action=self.trajectory.actions[t]
      local observation=self.trajectory.observations[t]
      self.dmodules[t]:reinforce(sum_reward)
      local delta_d=self.dmodules[t]:backward(self.states[t],nil)
      delta=delta+delta_d
      delta=self.rmodules[self.trajectory.actions[t-1]][t-1]:backward({self.states[t-1],self.trajectory.observations[t]},delta)[1]
    end
    self.dmodules[1]:reinforce(sum_reward)
    local delta_d=self.dmodules[1]:backward(self.states[1],nil)
    delta=delta+delta_d
    self.arguments.initial_recurrent_module:backward({self.arguments.initial_state,self.trajectory.observations[1]},delta)
             
    return -sum_reward,self.grad           
  end
end

function RecurrentPolicyGradient:new_episode(initial_observation,informations)
  self.trajectory=rltorch.Trajectory()
  self.states={}
  self.position=1
  self.last_sensor=self.sensor:process(initial_observation):clone()
  self.trajectory:push_observation(self.last_sensor)
  
  --coputing the first state. 
  self.states[1]=self.arguments.initial_recurrent_module:forward({self.arguments.initial_state,self.last_sensor})
end

function  RecurrentPolicyGradient:observe(observation)  
  self.last_sensor=self.sensor:process(observation):clone()
  self.trajectory:push_observation(self.last_sensor)
  
  self.states[self.position+1]=self.rmodules[self.trajectory.actions[self.position]][self.position]:forward({self.states[self.position],self.last_sensor})
  self.position=self.position+1  
end

function RecurrentPolicyGradient:feedback(reward)
  self.trajectory:push_feedback(reward)
end

function RecurrentPolicyGradient:sample()
  local out=self.dmodules[self.position]:forward(self.states[self.position])  
  local vmax,imax=out:max(2)
  self.trajectory:push_action(imax[1][1])
  return(imax[1][1])
end

function RecurrentPolicyGradient:end_episode(feedback)
  self.reward_trajectory=feedback*self.scaling_reward  
  local _,fs=self.optim(self.feval,self.params,self.optim_params) 
end

 
