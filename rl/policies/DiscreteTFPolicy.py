from rl.policies import Policy
import tensorflow as tf
import numpy as np

class DiscreteTFPolicy(Policy):
    # tensor_flow_model[input]=>input
    # tensor_flow_model[output]=>output
    # tensor_flow_model[init]=> initialization
    def __init__(self, observation_space, action_space, sensor, tensor_flow_model):
        Policy.__init__(self, observation_space, action_space)
        self.sensor=sensor
        self.policy_input=tensor_flow_model['input']
        self.policy_output=tf.nn.softmax(tensor_flow_model['output'])
        self.policy_init=tensor_flow_model['init']

        self.session=tf.Session()
        self.session.run(self.policy_init)

    def reset(self,initial_observation):
        self.observation=initial_observation

    def observe(self,observation):
        self.observation=observation


    def sample(self):
        x=self.sensor.process(self.observation)

        with tf.variable_scope('policy'):
            scores=self.session.run(self.policy_output,feed_dict={self.policy_input:x.reshape(1,-1)})
            sample=np.random.multinomial(1,scores[0])
            print(sample)
            action=np.where(sample==1)[0][0]
            return action



