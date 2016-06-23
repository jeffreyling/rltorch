 # Core Classes

The rltorch package is based on a few main classes.
* Environment: It describes the environment. Note that the environment also includes the sampling of the initial state
* Policy: It describes a policy (or agent) interacting with the environment
* Sensor: The sensor class aims at transforming observations (e.g states) coming from the environment to a policy. It allows one to test different views upon a same environment
* Space: This class describes spaces (for example discrete space, grid space, etc...) and is basically used to describe the action and observation spaces
* Trajectory: This class is used to store a complete trajectory in memory
* Trajectories: This class is used to store a set of trajectories in memory 

# Environment

The class is an abstract class used to implement new environments. An environment must contain the two variables:
* `action_space`: the space of the actions
* `observation_space`: the space of the observations

It is composed of the following methods:
* `__init(parameters)` : Initialization of a new enviroment. `parameters` is a table that contains the parameters of the environment. Note that the parameters are always stored in `self.parameters`
* `(observation, reward, done, informations) step(agent_action)`: It applies the `agent_action` to the environment and returns `(observation, reward, done, informations)`. `observation` is the new observation, `reward` is the feedback provided to the agent immediately after applying the action (if any). It is typically a scalar value (to maximize). `done` is true if the environment is in a final state. `informations` can be used to provide any other information to the agent (if needed)
* `observation reset()` : It resets the environment by sampling a new initial state, and returns the corresponding `observation`
*  `close()` : can be used at the end of the process to free the memory for example
* `render(...)`: can be used to visualize the state of the environment 

Note that the observation provided by the environment is updated through the `step` method, and thus has to be copied by the agent if it wants to keep memory of the trajectory

# Policy

This class describes an agent actiong in an environment. It is based on a `Sensor` (see below). It correponds to `P(a_t | sensor(o_t))` where `sensor(o_t)` is the observation of the environment throught the sensor. 

The methods are:
* `__init(observation_space,action_space,sensor)`: It initializes the policy (given a particular sensor if needed)
* `new_episode(initial_observation,informations)`: must be called at the begining of a trajectory (just after the Environment:reset() function). Note that `informations` can be used to give external information to the policy.
* `observe(observation)`: must be called before sampling a new action
* `sample()`: It samples an action given the last observed observation.
* `feedback(reward)`: It provides feedback (can be a scalar or any other structure depending on the nature of the policy) corresponding to the last sampled action
* `end_episode(feedback)`: must be called at the end of a trajectory. `feedback` corresponds to the feedback provided for the whole trajectory (e.g the total reward when using the policy gradient algorithm)


# Sensor

A sensor transforms an observation and is used by a policy. It allows one to test different views on the same environment. The basic work of a sensor is to transform an `observation` to a (1,n) torch Tensor (but many other cases can be imagined). It is based on the `process(observation)` method. Note that the result of `process` must be copied by the policy if it wants to keep a copy of the whole trajectory.

# Trajectory/Trajectories

A memory for one or many trajectories (simple classes, not essential). A trajectory is a sequence `(o_1,a_1,r_1,o_2,a_2,r_2,...,o_T)`



