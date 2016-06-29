
This package is a Reinforcement Learning package written in LUA for Torch. It main features are (for now):
* Different environments are provided, from classical RL environments, ATARI games, to special ones like the `multiclass classification environment` that casts a classification learning problem to a RL problem.
  * Sequential Acquisition Environment as described in `Gabriel Dulac-Arnold, Ludovic Denoyer, Philippe Preux, Patrick Gallinari: Sequential approaches for learning datum-wise sparse representations. Machine Learning 89(1-2): 87-122 (2012)`
* Different learning policies are provided:
  * Classic reward-based policies: Policy gradient, recurrent policy gradient, approximated q-learning with experience replay (also known as deep Q learning)
  * Imitation-based policies: Stochastic gradient-based imitation policy
  * Predictive policies: policies which goal is to predict an output (for example in order to make classification). 
* The different policies can be easily used with `openAI Gym` directly in python by using the `lutorpy` package

More features are planed:
* New environments: 
  * Text classification environment (sequential reading) as described in `Gabriel Dulac-Arnold, Ludovic Denoyer, Patrick Gallinari: Text Classification: A Sequential Reading Approach. ECIR 2011: 411-423`
  * Image classification with attention as described in `Gabriel Dulac-Arnold, Ludovic Denoyer, Nicolas Thome, Matthieu Cord, Patrick Gallinari: Sequentially Generated Instance-Dependent Image Representations for Classification. ICLR 2014`   
  * ...
* New learning policies

# Dependencies

Lua: 
* [Torch7](http://torch.ch/docs/getting-started.html#_)
* nn, dpnn
* logroll, json, alewrap, sys, paths, tds
```bash
luarocks install nn
luarocks install dpnn
luarocks install logroll
luarocks install json
luarocks install sys
luarocks install paths
luarocks install tds
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
  * [Classic Control Tasks](doc/env_classiccontrol.md): Classic control tasks
  * [Atari](doc/env_atari.md): Atari environments
  * [Classic Machine Learning](doc/env_classicmachinelearning.md): We also provide some environments that correspond to classical machine learning problems seen as RL environments (multiclass classification for now, one shot learning and structured output prediction for the future).
  *  * 
* [Tools](doc/tools.md): different tools

# OpenAI Gym

The interface with the open AI Gym package is explained [Here](doc/openai.md)

# Tutorials

The tutorials are avaialbe here: [Tutorials](doc/tutorials.md)

# FAQ

1. When installing Lutorpy, Luajit is not being detected.

Check that pkg-config can find luajit. The following should return at least one result:

```
pkg-config --list-all | grep luajit
```

If there are no results, then your `.pc` file for luajit is probably not in the right place. Try something like the following:

```
ln -s /path/to/torch/exe/luajit-rocks/luajit-2.0/etc/luajit.pc /usr/local/lib/pkgconfig/luajit.pc
```

2. Exception related to no display, such as `pyglet.canvas.xlib.NoSuchDisplayException: Cannot connect to "None"`.

OpenAI Gym needs some sort of display to record results. On ubuntu, you may try to install xvfb/asciinema. Then try running example.py like so:

```
xvfb-run -s "-screen 0 1400x900x24" python example.py
```

Author: Ludovic DENOYER -- The code is provided as if, some bugs may exist.....
