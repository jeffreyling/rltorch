 
require('rltorch')
env = rltorch.MountainCar_v0()
env:reset()
for i=1,1000 do
    env:render{mode="console"}
    --env:render{mode="qt"}
    local observation,reward,done,info=unpack(env:step(env.action_space:sample()))    
end