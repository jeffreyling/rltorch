
--- This environment implements a sequential sparse environment. Each trajectory corresponds to one data point (in a labeled set). At each timestep, the policy chooses which features to acquire (action). The environment provided the target at each timestep. The observation is only the features value.


local SparseSequentialLearning_v0 = torch.class('rltorch.SparseSequentialLearning_v0','rltorch.Environment'); 
  
 
----
-- Parameters:
----- training_examples: a (N x n) tensor where N is the number of training examples and n the dimension of the input space
----- training_labels : a (N x 1) tensor, where each value corresponds to the true label of each example
----- testing_examples (optionnal) : a (N' x n) tensor where N' is the number of testing examples. If (self.test==true) then the environment will generate a trajectory of all the testing examples.
----- testing_labels (optionnal) : corresponding labels
function SparseSequentialLearning_v0:__init(parameters)        
        self.parameters=parameters        
        assert(self.parameters.training_examples~=nil)
        assert(self.parameters.training_labels~=nil)
        self.n=self.parameters.training_examples:size(2)
        self.action_space = rltorch.Discrete(self.n)
        
        local vmin=torch.Tensor({self.parameters.training_examples:min()})
        local vmax=torch.Tensor({self.parameters.training_examples:max()})
        self.observation_space = rltorch.Box(vmin,vmax)     
        self.test=false
        self.current_data_idx=0
end
 
--- the 4-th element of the return contains the true category of the last example. The reward is a zero/one reward
function SparseSequentialLearning_v0:step(agent_action)  
  --- TESTING MODE
        if (self.test) then          
          local true_category=self.parameters.testing_labels[self.current_data_idx][1]          
          local feedback={target=torch.Tensor({true_category})}         
          local x=self.parameters.testing_examples[self.current_data_idx][agent_action]
          local obs=torch.Tensor({x})
          return {obs,nil,false,feedback}
        else ---- TRAINING MODEL
          local true_category=self.parameters.training_labels[self.current_data_idx][1]
          local feedback={target=torch.Tensor({true_category})}
          local x=self.parameters.training_examples[self.current_data_idx][agent_action]
          local obs=torch.Tensor({x})
          return {obs,nil,false,feedback}
        end
end


function SparseSequentialLearning_v0:reset(use_test)
    if (use_test) then
      if (self.test==true) then self.current_data_idx=self.current_data_idx+1 else self.current_data_idx=1 end
      if (self.current_data_idx>self.parameters.testing_examples:size(1)) then self.current_data_idx=1 end
      self.test=true
    else
      self.test=false
      self.current_data_idx=math.random(self.parameters.training_examples:size(1))
    end
    return(torch.Tensor(1):fill(0))
end 

function SparseSequentialLearning_v0:close()
end

function SparseSequentialLearning_v0:render(arg)
  if (arg.mode=="console") then
  end
end

 
