 require 'rl'
require 'nn'
 
package.path = '/home/denoyer/alewrap/?/init.lua;' .. package.path
local alewrap = require 'alewrap'
DISCOUNT_FACTOR=1.0

WIDTH=160
HEIGHT=210

environment=rl.ALEEnvironment("../roms/seaquest.bin",WIDTH,HEIGHT)
factory=rl.ALEFactory()
sensor=rl.ALESensor(WIDTH,HEIGHT)
domain=environment:getDomain()

random_policy=rl.RandomPolicy(#domain)

sampler_parameters={
  environment=environment,
  factory=factory,
  policy=random_policy,
  sensor=sensor,
  size_max_trajectory=10000
}

sampler=rl.TrajectorySampler(sampler_parameters,DISCOUNT_FACTOR)

t=sampler:sample()
torch.save("coucou",t)
t2=torch.load("coucou")
print(t2)