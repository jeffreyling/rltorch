from rl.policies import Policy
import tensorflow as tf
import numpy as np

from rl.tools import Trajectories


class DiscretePolicyGradient(Policy):
    #args:
    #   size_batch:
    def __init__(self, observation_space, action_space, sensor, tensor_flow_model,**args):
        Policy.__init__(self, observation_space, action_space)

        assert('size_batch' in args),"'size_batch' must be defined"
        self.memory=Trajectories(self.observation_space,self.action_space)

        self.sensor=sensor
        self.policy_input=tensor_flow_model['input']
        self.policy_output_scores=tensor_flow_model['output']
        self.policy_output_probabilities=tf.nn.softmax(tensor_flow_model['output'])
        self.policy_init=tensor_flow_model['init']

        #Creating gradient network
        #The gradient networks takes a sequence of observations and actions as input + reward and return the sum of log proba * R. It also takes the sequence length
        self.g_observations=tf.placeholder(tf.float32, [None, num_steps])



        self.session=tf.Session()
        self.session.run(self.policy_init)

    def reset(self, initial_observation):
        self.memory.new_trajectory()
        self.memory.push_observation(initial_observation)
        self.observation = initial_observation

    def observe(self, observation):
        self.memory.push_observation(observation)
        self.observation=observation

    def feedback(self, reward):
        self.memory.push_reward(reward)

    def sample(self):
        x=self.sensor.process(self.observation)

        with tf.variable_scope('policy'):
            scores=self.session.run(self.policy_output_probabilities,feed_dict={self.policy_input:x.reshape(1,-1)})
            sample=np.random.multinomial(1,scores[0])
            print(sample)
            action=np.where(sample==1)[0][0]
            self.memory.push_action(action)
            return action

    def end(self):
        if (self.memory.size()<self.size_batch):
            return

        #Here, we have to learn !!!
        print("Go for learning....")

        #if self.size_batch>1 then we assume that all the trajectories have the same length....


    def reinforce_graph(self,input,scores):
        with tf.variable_scope('reinforce'):
            length=tf.placeholder(tf.int32)
            position=tf.constant(0)

            def condition(l):
                return(l[0]<l[1])




            all_observations=tf.placeholder(tf.float32,[None,])



