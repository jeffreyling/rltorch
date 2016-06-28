require ('optim') 
require('rltorch')

MAX_LENGTH=100
NB_TRAJECTORIES=10000

env = rltorch.CartPole_v0()
math.randomseed(os.time())
sensor=rltorch.BatchVectorSensor(env.observation_space)

local size_input=sensor:size()
local nb_actions=env.action_space.n

-- Creating the policy module
local module_policy=nn.Sequential():add(nn.Linear(size_input,nb_actions)):add(nn.SoftMax()):add(nn.ReinforceCategorical())
module_policy:reset(0.01)

local arguments={
    policy_module = module_policy,
    max_trajectory_size = MAX_LENGTH,
    optim=optim.adam,
    optim_params= {
        learningRate =  0.01  
      },
    scaling_reward=1.0/MAX_LENGTH,
    size_memory_for_bias=100
  }
policy=rltorch.PolicyGradient(env.observation_space,env.action_space,sensor,arguments)

local ptools={}; 
  ptools.discount_factor=1
  ptools.size_max_trajectories=MAX_LENGTH
  ptools.nb_trajectories=NB_TRAJECTORIES
--  ptools.render_parameters={mode="qt",fps=30}
--  ptools.display_every=100
  
log=rltorch.ExperimentLogConsole()
rltorch.RLTools():monitorReward(environment,policy,log,ptools)