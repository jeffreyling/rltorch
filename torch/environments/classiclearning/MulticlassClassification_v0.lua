
--- This environment implements a classical multiclass classification problem: each training example is returned at random at each timstep. The feedback can be a reward (0/1 loss), or the category of the example depending on the init parameters

local MulticlassClassification_v0 = torch.class('rltorch.MulticlassClassification_v0','rltorch.Environment'); 
  
 
----
-- Parameters:
----- training_examples: a (N x n) tensor where N is the number of training examples and n the dimension of the input space
----- training_labels : a (N x 1) tensor, where each value corresponds to the true label of each example
----- testing_examples (optionnal) : a (N' x n) tensor where N' is the number of testing examples. If (self.test==true) then the environment will generate a trajectory of all the testing examples.
----- testing_labels (optionnal) : corresponding labels
function MulticlassClassification_v0:__init(parameters)        
        self.parameters=parameters        
        assert(self.parameters.training_examples~=nil)
        assert(self.parameters.training_labels~=nil)
        local vmax,imax=self.parameters.training_labels:max(1)
        self.action_space = rltorch.Discrete(vmax[1][1])
        self.n=self.parameters.training_examples:size(2)
        
        local vmin=self.parameters.training_examples:min(1)[1]:reshape(self.n)
        local vmax=self.parameters.training_examples:max(1)[1]:reshape(self.n)
        self.observation_space = rltorch.Box(vmin,vmax)     
        self.test=false
end
 
--- the 4-th element of the return contains the true category of the last example. The reward is a zero/one reward
function MulticlassClassification_v0:step(agent_action)  
  --- TESTING MODE
        if (self.test) then          
          local true_category=self.parameters.testing_labels[self.last_index][1]
          
          local feedback={true_action=true_category}
          local reward=0          
          if (true_category==agent_action) then reward=1 end
          
          self.last_index=self.last_index+1
          if (self.last_index>self.parameters.testing_examples:size(1)) then 
            return {nil,reward,true,feedback}
          else
            return {self.parameters.testing_examples[self.last_index],reward,false,feedback}
          end
        else ---- TRAINING MODEL
          local true_category=self.parameters.training_labels[self.last_index][1]
          local feedback={true_action=true_category}
          local reward=0          
          if (true_category==agent_action) then reward=1 end
          
          self.last_index=math.random(self.n)
          return {self.parameters.training_examples[self.last_index],reward,false,feedback}
        end
end


function MulticlassClassification_v0:reset(use_test)
    if (use_test) then
      self.test=true
      self.last_index=1
      return(self.parameters.testing_examples[self.last_index])
    else
      self.test=false
      self.last_index=math.random(self.n)
      return(self.parameters.training_examples[self.last_index])
    end
  
end 

function MulticlassClassification_v0:close()
end

function MulticlassClassification_v0:render(arg)
  if (arg.mode=="console") then
    local mode="train"
    if (self.test) then model="test" end
    print("Mode = "..mode.." : index = "..self.last_index)
  end
end

 
