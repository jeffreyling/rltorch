# Different Tools 

# Class RLTools

This class provides different tools for RL

`function RLTools:monitorReward(environment,policy,log,parameters)`: to estimate the quality of a policy based on rewards

This function launches nb_trajectories trajectories and save the total discounted reward + trajectories lengths in a log (see tutorials/t2_policygradient_bis.lua). The parameters are:
* `policy` : the policy
* `environment` : the environment
* `log` : the log object (see `log` directory)
* `parameters` (table):
  * `discount_factor` : the discount_factor (deault 1)
  * `nb_trajectories` : the number of trajectories (default 1000)
  * `size_max_trajectories` : the maximum size of each trajectory (default nil => + infinity)
  * `render_parameters` (optional) : if one wants to render the environment, then you have to provide the render parameters
  * `display_every` (optional) : display the reward evry n iterations

The policy receives the immediate reward at each timestep, and the discounted sum of rewards at the end of the episode.


# Class RLFile

Different file tools.

* `function RLFile:read_libsvm(filename)`
Read a libsvm file and returns a matrix of data and a matrix of labels. Labels are reindexed

* `function RLFile:split_train_test(data,labels,proportion_train)`
It generates a split (train and test) over  data and labels matrices, giben a particular proportion to keep in train. 



