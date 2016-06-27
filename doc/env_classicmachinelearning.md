# MulticlassClassification_v0
(see t6_multiclass_classification.lua)

This environment simulate a classical iterative training procedure for multiclass classification problems where one provides a training and testing set. 

When initializing the environment, you have to provide the following parameters:
* `training_examples` is a (n x N) matrix where n is the number of training examples, and N the dimension of the input space
* `training_labels` is a (n x 1) matrix where each value is the label (int) of the corresponding example. Labels must be between 1 and C
* `testing_examples` (optionnal) is a (n' x N) matrix where n' is the number of testing examples
* `testing_labels` (optionnal) is a (n' x 1) matrix
* `zero_one_reward`: if `true` then the environment returns 1 if action corresponds to the good label, and 0 elsewhere. if `false` (classical RL problem), the environment returns the true label at each timestep, and must thus be handled with specific policies that are not based on rewards

When using `reset`, you have to say if you want to use (`reset(true)`) or not (`reset(false)`) the testing mode: 
* During the training mode, training examples are sampled uniformly, and the environment never stops
* During the testing model, testing examples are sampled from the first one to the last one (one trajectory is an iteration over all the testing examples)

 
