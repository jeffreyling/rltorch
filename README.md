
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
  * [Classic Control Tasks](doc/env_classiccontrol.md): Classic control tasks
  * [Atari](doc/env_atari.md): Atari environments
  * [Classic Machine Learning](doc/env_classicmachinelearning.md): We also provide some environments that correspond to classical machine learning problems seen as RL environments (multiclass classification for now, one shot learning and structured output prediction for the future)

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

# Other Information

[TODO](TODO.md) : What will happen next

Author: Ludovic DENOYER
