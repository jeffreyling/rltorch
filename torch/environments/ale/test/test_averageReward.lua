require 'rl'
require 'nn'
 
package.path = '/home/denoyer/alewrap/?/init.lua;' .. package.path
local alewrap = require 'alewrap'
DISCOUNT_FACTOR=1.0

environment=rl.ALEEnvironment("../roms/seaquest.bin")
factory=rl.ALEFactory()
sensor=rl.ALESensor(160,210)
domain=environment:getDomain()
print(domain)


random_policy=rl.RandomPolicy(#domain)

sampler_parameters={
  environment=environment,
  factory=factory,
  policy=random_policy,
  sensor=sensor,
  size_max_trajectory=10000
}

sampler=rl.DiscountedRewardSampler(sampler_parameters,DISCOUNT_FACTOR)
while(true) do
  print("Avg Discounted Reward over 1 samples = ",sampler:MonteCarloMean(1))
end
