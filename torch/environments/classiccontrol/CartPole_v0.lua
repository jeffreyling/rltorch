
local CartPole_v0 = torch.class('rltorch.CartPole_v0','rltorch.Environment'); 
  
 
--- Initialize the environment
function CartPole_v0:__init()
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5 -- actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 10.0
        self.tau = 0.02  -- seconds between state updates

        -- Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        -- Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        local high = torch.Tensor({self.x_threshold, math.huge, self.theta_threshold_radians * 2, math.huge])
        self.action_space = rltorch.Discrete(2)
        self.observation_space = rltorch.Box(-high, high)
end
 
---Update the environment given that one agent has chosen one action
-- @params agent_action the action of the agent
-- @returns observation,reward,done,info
--- Here action is 1 2 or 3
function CartPole_v0:step(agent_action)  
        self.action = agent_action
        assert(agent_action==1 or action==2), "Invalid action")
        local x=state[1]
        local x_dot=state[2]
        local theta=state[3] 
        local theta_dot = state[4]
        
        force = self.force_mag if action==1 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta* temp) / (self.length * (4.0/3.0 - self.masspole * costheta * costheta / self.total_mass))
        xacc  = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        x  = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
        self.state = (x,x_dot,theta,theta_dot)
        done =  x < -self.x_threshold \
                or x > self.x_threshold \
                or theta < -self.theta_threshold_radians \
                or theta > self.theta_threshold_radians
        done = bool(done)

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state), reward, done, {}end

---- Returns the initial domain 
-- @return the action domain
function CartPole_v0:reset()
   self.state = torch.Tensor({math.random()*(-0.4+0.6)-0.4, 0})
   return(self.state)
end 

--- Tells if we are in a terminal state or not
function CartPole_v0:close()
  
end

--- Clone the environment
function CartPole_v0:render(arg)
  if (arg.mode=="console") then
    print("Position = "..self.state[1].." / Speed = "..self.state[2])
  end
end
