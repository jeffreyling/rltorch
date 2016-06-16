
require 'image'

package.path = '/home/denoyer/alewrap/?/init.lua;' .. package.path
local alewrap = require 'alewrap'

local function parseArgs()
    local cmd = torch.CmdLine()
    cmd:option('-rom', '../roms/breakout.bin', 'ROM to use')

    return cmd:parse(arg)
end


local function main()
    local options = parseArgs()
    -- Starting the env.
    local env = alewrap.createEnv(options.rom, {})
	for k,v in pairs(env:getEnvSpec()) do
		print(k)
		print(v)
	end
	    local observations = env:envStart()
    local reward

    local win = nil
    while true do
        -- Displaying the atari screen.
        local obs = observations[1]
--	print(obs)
        win = image.display({image=env:getRgbFromPalette(obs), win=win})

	

        -- Taking an action.
        local action = torch.random(0, 17)
        local actions = {torch.Tensor({action})}
        reward, observations = env:envStep(actions)
        if reward ~= 0 then
            print("reward:", reward)
        end
    end
end

main()
