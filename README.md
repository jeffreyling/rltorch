
This package is a basic Reinforcement Learning package written in LUA for Torch. It implements some simple environments and learning policies (Policy Gradient and Deep Q Learning). It also can be easily used with the OpenAI Gym package by using lutorpy (example given in the opeanaigym directory).

Tutorials are provided in the tutorials directory

# Dependencies

Lua: 
* [Torch7](http://torch.ch/docs/getting-started.html#_)
* nn, dpnn
* logroll, json, alewrap
```bash
luarocks install nn
luarocks install dpnn
luarocks install logroll
luarocks install json
git clone https://github.com/deepmind/xitari.git && cd xitari && luarocks make && cd .. && rm -rf xitari
git clone https://github.com/deepmind/alewrap.git && cd alewrap && luarocks make && cd .. && rm -rf alewrap
```

For using openAI Gym:
* openai gym
* lutorpy

# Installation

* `cd torch && luarocks make`
* Install [lutorpy](https://github.com/imodpasteur/lutorpy) and [OpenAI Gym](https://gym.openai.com/)
* lauch the python script (example.py)

# Documentation

The package if composed of these different elements:
* [Core](doc/core.md): the core classes
* [Sensors](doc/sensors.md): the different sensors
* [Policies](doc/policies.md): different (learning) policies
* [Environments](doc/environments.md): different environments

The interface with the open AI Gym package is explained [Here](doc/openai.md)


Author: Ludovic DENOYER

