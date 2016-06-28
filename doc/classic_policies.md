# Classic Policies

Policies that are driven by a reward signal. Depending on the policy, the reward can be provided at each timestep, or at the end of each episode

# Policy Gradient
 (see `tutorials/t2_policygradient.lua` example)

An implementation of the REINFORCE algorithm for discrete action spaces. When creating the policy, one has to provide the following values:
* `policy_module`: a reinforced module (see package dpnn) that takes a (1,n) observation as an input an returns a (1,A) vector where `A` is the number of possible actions. The returned vector is a onehot vector with a 1 for the sampled action
* `max_trajectory_size`: the maximum size of the trajectories. It is used to know how many times the base module has to be copied in memory
* `optim`: the `optim` method (see `optim` ptorch package)
* `optim_param`: the parameters used for the `optim` method

Optionnal parameters:
* `size_memory_for_bias`: the number of trajectories used for computing the average reward obtained by the policy. This average value is used in the reinforce algorithm and substracted to the total observed reward.
* `scaling_reward`: the scaling factor for the received reward

# RecurrentPolicyGradient
(see tutorials/t5_recurrentpolicygradient.lua)

This class provides an implementation of the REINFORCE algorithm using recurrent neural network. The principle is the following:
* The neural network involves a `N` dimensionnal latent space
* At the begining of a neuw trajectory, the network state is `z_0` which is proided by the user when creating the policy
* Given the first observation, the state `z_1` is computed using the `initial_recurrent_module` which transforms a {(1,N,(1,O)} pair of tensors to a (1,N) tensor where `O` is the size of the observation space (after processing by the sensor)
* Then, for each timestep `t`, the action `a_t` is sampled by using the `policy_module` applied on `z_t`. The `policy_module` is based on the `dpnn` package using the reinforced modules
* The next state `z_{t+1}` is computed based on the chosen action and the obtained observation through the `recurrent_module[a_t]` which takes as an input a `{(1,N),(1,O)}` pair of tensors and computes `z_{t+1}`


# DeepQPolicy
 (see `tutorials/t4_deepqlearning.lua`)

An approximated Q-Learning with Experience Replay. (Note: if anybody can check this code, it could be nice....). The parameters are:
* `policy_module`: a torch nn module  `(1,n) -> (1,A)` providing the score of the Q-function for all the possible actions
* `optim` and `optim_params`: the `optim` function to use
* `size_memory`: the size of the memory for the experience replay
* `size_minibatch`: the number of examples used at each gradient iteration. The examples are uniformly sampled in the memory of the policy
* `discount_factor`: the discount factor since DeepQ aims at maximizing a discounted reward
* `epsilon_greedy`: when `self.train=true`, the policy samples the action given an espilon greedy policy (where `epsilon_greedy` is the value of epsilon). When `self.train=false`, then the action is chosen using the max of the Q-value
* `scaling_reward`: the scaling factor for the received reward

Note that, here, the SGD step is made at the end of each trajectory (i.e in the `end_episode` function) 
