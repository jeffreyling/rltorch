import gym

import rl.tools as rltools
import rl.policies as rlpolicies
import rl.policies.learning as rlpolicieslearning
import rl.policies.sensors as rlsensors
import tensorflow as tf

env = gym.make('Acrobot-v0')
sensor=rlsensors.SensorImageToVector(env.observation_space)

#The model
model={}
with tf.variable_scope('my_model') and tf.device('/cpu:0'):
    STDV=0.01
    policy_input = tf.placeholder(tf.float32, [None, sensor.size()])
    w = tf.Variable(tf.random_normal([sensor.size(), env.action_space.n], stddev=STDV))
    b = tf.Variable(tf.random_normal([env.action_space.n], stddev=STDV))
    policy_output = tf.matmul(policy_input, w) + b
    model['input']=policy_input
    model['output']=policy_output
    model['init']=tf.initialize_all_variables()

#The policy
policy=rlpolicieslearning.DiscretePolicyGradient(env.observation_space,env.action_space,sensor,model,size_batch=5)

for i_episode in range(10):
    io=env.reset()
    policy.reset(io)
    for t in range(1000):
        env.render()
        action = policy.sample()
        observation, reward, done, info = env.step(action)
        policy.feedback(reward)
        policy.observe(observation)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
    policy.end()
