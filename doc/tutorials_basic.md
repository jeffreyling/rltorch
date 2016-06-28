# Basic Tutorials

## Tutorial : Testing an environment
(see tutorials/t0_running.lua)

* Creation of a new environment: 

```lua
    env=rltorch.MountainCar_v0()
    print(env.observation_space)
    print(env.action_space)
```

* Sample 1 000 states from this environment choosing the actions with a uniform distribution!

```lua
    for i=1,1000 do
        env:render{mode="console"}
        local observation,reward,done,info=unpack(env:step(env.action_space:sample()))    
    end
```

Note that, in many environments, one can use `env:render{mode="qt",fps="30"}` if one wants to view the environment using Qt. `fps` is the max number of frames per second. In this code, the action is sampled from the `action_space` using the `Space:sample()` method

# Tutorial 1:  A (random) agent interacting with an environment
(see tutorials/t1_policy.lua)

*  Create the environment
```lua
    env = rltorch.MountainCar_v0()
```

* Create the agent (or policy)
```lua
    policy=rltorch.RandomPolicy(env.observation_space,env.action_space)
```

* Sample one trajectory:
  * First, sample the initial state, and transmit the initial observation to the agent

```lua
    local initial_observation=env:reset()
    policy:new_episode(env:reset())  
```
  * Second, the main loop:

```lua 
    local total_reward=0 
    while(true) do
      env:render{mode="console"}      
      --env:render{mode="qt"}      

      local action=policy:sample()  -- sample one action based on the policy
      local observation,reward,done,info=unpack(env:step(action))  -- apply the action
      policy:feedback(reward) -- transmit the (immediate) reward to the policy
      policy:observe(observation)      
      total_reward=total_reward+reward -- update the total reward for this trajectory

      --- Leave the loop if the environment is in a final state
      if (done) then
        break
      end
    end
    policy:end_episode(total_reward) -- Transmit the total trajectory reward to the policy
   env:close()
```

Note that, depending on the policy, sometimes the immediate feedback will be necessary (DeepQLearning for example), sometimes only the trajectory feedback will be enough (PolicyGradient)

