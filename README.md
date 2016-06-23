
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


WARNING : If you use an openAI Gym ATARI environment, a new sensor must be developed: it will be avaiable in the next few days (since openAI and alewrap do not store the ATARI images in the same format)

Author: Ludovic DENOYER

