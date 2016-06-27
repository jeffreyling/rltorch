require 'nn'
require 'dpnn'  
  
  --- A policy based on Deep Q Learning (using a epsilon-greedy exploration)
local DeepQPolicy = torch.class('rltorch.DeepQPolicy','rltorch.Policy'); 

--- the policy_module is a R^n -> nb_actions >sampling vector
--- ARGUMENTS= 
----- policy_module = the policy module (takes a B*n matrix to a B*A vector with one q value for each action, B being the size of the minibatch)
----- optim = the optim method
----- optim_params = the optim state
----- size_minibatch = the size of the mini-batches
----- size_memory = the size of the memory for experience reply
----- discount_factor = the discount_factor
----- epsilon_greedy = the value of epsilon in epsilon greedy
----- scaling_reward = the reward sclaing value
function DeepQPolicy:__init(observation_space,action_space,sensor,arguments)
  rltorch.Policy.__init(self,observation_space,action_space) 
  self.sensor=sensor
  self.arguments=arguments
  
  assert(arguments.policy_module~=nil)
  assert(arguments.optim~=nil)
  assert(arguments.optim_params~=nil)
  assert(arguments.size_minibatch~=nil)
  assert(arguments.size_memory~=nil)
  assert(arguments.discount_factor~=nil)
  assert(arguments.epsilon_greedy~=nil)
  assert(arguments.scaling_reward~=nil)
  
  self.memory={}
  self.memory_position=0
  self.memory_size=0
  
  self.optim=arguments.optim
  self.optim_params=arguments.optim_params
  self.is_training=true
  
  if (arguments.scaling_reward==nil) then self.scaling_reward=1 else self.scaling_reward=arguments.scaling_reward end 
  
  self.policy_module=arguments.policy_module
  self.max_trajectory_size=arguments.max_trajectory_size
  self.train=true
  self:init()
end

function DeepQPolicy:init()    
  self.params, self.grad = rltorch.ModelsUtils():combine_all_parameters(self.policy_module) 
  self.loss=nn.MSECriterion()
  
  self.tensor_observation_t=torch.Tensor(self.arguments.size_minibatch,self.sensor:size()) -- the tensor with the observations for the minibatch
  self.tensor_objective=torch.Tensor(self.arguments.size_minibatch,self.action_space.n) -- the objective of learning
  self.tensor_observation_t_plus_one=torch.Tensor(self.arguments.size_minibatch,self.sensor:size()) -- the tensor with the observations for the minibatch
  
  self.feval = function(params_new)
    if self.params ~= params_new then
        self.params:copy(params_new)
    end
    
    self.grad:zero()
    
    --- bulding the minibatch
    local idxs={}
    for b=1,self.arguments.size_minibatch do
      local idx=math.random(self.memory_size)
      while(self.memory[idx].observation_plus_one==nil) do idx=math.random(self.memory_size) end
      idxs[b]=idx
--      print(self.memory[idx])
      self.tensor_observation_t[b]:copy(self.memory[idx].observation[1])
      self.tensor_observation_t_plus_one[b]:copy(self.memory[idx].observation_plus_one[1])      
    end
    
    local vmax,imax=self.policy_module:forward(self.tensor_observation_t_plus_one):max(2)
    
    local out=self.policy_module:forward(self.tensor_observation_t)
    self.tensor_objective:copy(out)
    
    for b=1,self.arguments.size_minibatch do
      local action=self.memory[idxs[b]].action
      local reward=self.memory[idxs[b]].reward*self.arguments.scaling_reward
      if (not self.memory[idxs[b]].done) then
          reward=reward+vmax[b][1]*self.arguments.discount_factor
      else
      
    end
    -- print(reward)
      self.tensor_objective[b][action]=reward
    end
    
    
    local loss=self.loss:forward(out,self.tensor_objective)
    local delta=self.loss:backward(out,self.tensor_objective)
    self.policy_module:backward(self.tensor_observation_t,delta)
    
    return loss,self.grad           
  end
end

function DeepQPolicy:chooseMemoryCell(ob)
--  if (self.memory_size==self.arguments.size_memory) then
--    local _,fs=self.optim(self.feval,self.params,self.optim_params)
--  end  
  
  if (self.memory_size<self.arguments.size_memory) then
    self.memory_position=self.memory_position+1
    self.memory_size=self.memory_size+1
  else
    self.memory_position=math.random(self.memory_size)
    while(self.last_memory_position==self.memory_position) do self.memory_position=math.random(self.memory_size) end
  end
  self.memory[self.memory_position]={}
  self.memory[self.memory_position].observation=ob  
end  

function DeepQPolicy:new_episode(initial_observation,informations)
  self.last_sensor=self.sensor:process(initial_observation):clone()
  self:chooseMemoryCell(self.last_sensor)
end

function  DeepQPolicy:observe(observation)  
  self.last_sensor=self.sensor:process(observation):clone()
  self.last_memory_position=self.memory_position
  self.memory[self.memory_position].observation_plus_one=self.last_sensor
  
  self:chooseMemoryCell(self.last_sensor)
end

function DeepQPolicy:feedback(reward)
  self.memory[self.memory_position].reward=reward
  self.memory[self.memory_position].done=false  
end

function DeepQPolicy:sample()
  if ((math.random()<self.arguments.epsilon_greedy) and (self.train)) then
    action_taken=math.random(self.action_space.n)
  else
    local out=self.policy_module:forward(self.last_sensor)
    local vmax,imax=out:max(2)
    action_taken=imax[1][1]
  end
  
  self.memory[self.memory_position].action = action_taken
  return(action_taken)
end

function DeepQPolicy:end_episode(feedback)
  self.memory[self.last_memory_position].done=true

  if (self.memory_size==self.arguments.size_memory) then
    if (self.train) then local _,fs=self.optim(self.feval,self.params,self.optim_params) end
  end  


end

 
