# Predictive Policies

A predictive policy is a policy that aims at providing a final predictions on which a loss will be computed (and not only a reward). The interest is that these policies will benefit from a reicher signal

# PredictiveRecurrentPolicyGradient

This policy aims at predicting ''at the end of the trajectory''. It declares a `predict` method that computes a prediction at each timestep. It is also based on a `Criterion` that will provide a rich feedback signal than a simple reward. It may thus only be used will environment that provides a particular target at the end of a trajectory. 

The objective loss is not to maximize a reward, but to minimize a loss between the prediction and the target. In comparison to `RecurrentPolicyGradient`, a `PredictiveRecurrentPolicyGradient` also uses a `predictive_module` which goals is to compute a prediction based on the current latent space. The policy also needs a `criterion` that will be used to evaluate the quality of the prediction. 



