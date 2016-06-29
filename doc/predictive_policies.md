# Predictive Policies

A predictive policy is a policy that aims at providing a final predictions on which a loss will be computed (and not only a reward). The interest is that these policies will benefit from a reicher signal

# PredictiveRecurrentPolicyGradient

This policy aims at predicting ''at the end of the trajectory''. It declares a `predict` method that computes the prediction at each timestep. It is also based on a `Criterion` that will provide a rich feedback signal. It may thus only be used will environment that provide a particular target at the end of a trajectory. 

... more to come

