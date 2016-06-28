#Imitation Policies

Imitation policies are not driven by a reward, but by a `true action` which is the action to imitate. To be used, the environment must provide a `{true_action=...}` at each timestep (see `MuticlassClassificationEnvironment` for an example of an environment that can be used with classical policies since it provides a reward, but also with imitation policies)

# StochasticGradientImitationPoliciy

This policy implements a simple SGD based on a NLL Criterion. The SGD step is made when calling `end_episode` over all the observations of the current (ended) trajectory. The building parameters are:
* `policy_module` = the policy module (takes a 1*n matrix to a 1*n vector of scores, one score for each action. The softmax is applied by the current class and does not need to be included in the module)
* `optim` = the optim method (e.g optim.adam)
* `optim_params` = the optim initial state 


