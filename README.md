
This package is a basic Reinforcement Learning package written in LUA for Torch. It implements some simple environments and learning policies (Policy Gradient and Deep Q Learning). It also can be easily used with the OpenAI Gym package by using lutorpy (example given in the opeanaigym directory).

Tutorials are provided in the tutorials directory

# Dependencies

Lua: 
* Torch7
* nn, dpnn
* logroll, json, alewrap

For using openAI Gym:
* openai gym
* lutorpy

# Installation

* In the torch directory: luarocks make
* Install lutorpy and open AI
* lauch the python script (example.py)


WARNING : If you use an openAI Gym ATARI environment, a new sensor must be developped: it will be avaiable in the next few days (since openAI and alewrap do not store the ATARI images in the same format)

Author: Ludovic DENOYER

