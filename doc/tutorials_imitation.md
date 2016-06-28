#Imitation Tutorials

How can we learn to imitate ? 

Here, we will use environments that (in adition to a reward) provide a `true_action` information.

# Tutorial 7 : Learning by Imitation -- Multiclass Classification by SGD

This tutorial is very close to the tutorial number 6 since it involves the same environment (`MuticlassClassificationEnvironment`) which can be used for both reward-based policies, and also imitation policies.

First, we create the environment
```lua
local PROPORTION_TRAIN=0.5
local data,labels = unpack(rltorch.RLFile():read_libsvm('datasets/breast-cancer_scale'))
local parameters={}
parameters.training_examples, parameters.training_labels,parameters.testing_examples,parameters.testing_labels = unpack(rltorch.RLFile():split_train_test(data,labels,PROPORTION_TRAIN))

env = rltorch.MulticlassClassification_v0(parameters)
sensor=rltorch.BatchVectorSensor(env.observation_space)
--sensor=rltorch.TilingSensor2D(env.observation_space,30,30)

local size_input=sensor:size()
local nb_actions=env.action_space.n
```

Then, we create the policy based on a linear module that computes one score for each possible action
```lua
local module_policy=nn.Sequential():add(nn.Linear(size_input,nb_actions))
module_policy:reset(0.01)

local arguments={
    policy_module = module_policy,
    optim=optim.adam,
    optim_params= {
        learningRate =  0.01  
      },    
  }
  
policy=rltorch.StochasticGradientImitationPolicy(env.observation_space,env.action_space,sensor,arguments)
```

Now, we keep the same loop than tutorial 6 (for both evaluation on the testing and training sets). The only difference is that we provide, as a feedback, the true action given by the environment. The difference is in these two lines:
```lua
      local observation,reward,done,feedback=unpack(env:step(action))    
      policy:feedback(feedback.true_action) 
```
instead of 
```lua
      local observation,reward,done,feedback=unpack(env:step(action))    
      policy:feedback(reward)
```
