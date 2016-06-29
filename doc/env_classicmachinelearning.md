# MulticlassClassification_v0
(see t6_multiclass_classification.lua)

This environment simulates a classical iterative training procedure for multiclass classification problems where one provides a training and testing set. At each timestep, the agent receive a new data point. It then has to predict the category of this point (action). As a feedback, the agent receives a 0/1 reward, the fourth term of the return (`step`) also contains the true label

When initializing the environment, you have to provide the following parameters:
* `training_examples` is a (n x N) matrix where n is the number of training examples, and N the dimension of the input space
* `training_labels` is a (n x 1) matrix where each value is the label (int) of the corresponding example. Labels must be between 1 and C
* `testing_examples` (optionnal) is a (n' x N) matrix where n' is the number of testing examples
* `testing_labels` (optionnal) is a (n' x 1) matrix

When using `reset`, you have to say if you want to use (`reset(true)`) or not (`reset(false)`) the testing mode: 
* During the training mode, training examples are sampled uniformly, and the environment never stops
* During the testing model, testing examples are sampled from the first one to the last one (one trajectory is an iteration over all the testing examples)

# SparseSequentialLearning_v0 
(see t8_predictivepolicy.lua)

This environment implements the MDP described in `Gabriel Dulac-Arnold, Ludovic Denoyer, Philippe Preux, Patrick Gallinari:Sequential approaches for learning datum-wise sparse representations. Machine Learning 89(1-2): 87-122 (2012)`:
* It is based on both a training and testing dataset with associated labels
* Each new trajectory is based on a randomly chosen training example. Each action corresponds to a features to acquire. The observation is the value of the acquired features. 
* At each timestep, the environment returns a `feedback.target` value that corresponds to the true category of the corresponding example. No reward is provided!!
* when using `reset(use_test)`, the `use_test` flag says if the new state is based on a testing or training example. 

This environment can be used with Predictive Policies



