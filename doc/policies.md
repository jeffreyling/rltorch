# Policies

The different (learning) policies implemented in the platform. We distinguish different types of policies:
* [Classic Policies](classic_policies.md) correspond to classic policies (policy gradient, deepq learning, ...)
* [Imitation Policies](imitation_policies.md) correspond to policies where the environment provide, at each step, the action made by a supervisor that the agent will try to imitate. 
* [Specific Policies] are policies developped for particular environments

Note that the package also provide simple policies as explained below.

# Random Policy

A random uniform policy for discrete action spaces

