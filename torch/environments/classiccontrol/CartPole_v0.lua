
local CartPole_v0 = torch.class('rltorch.CartPole_v0','rltorch.Environment'); 
  
 
function CartPole_v0:__init(batch_size)
        if batch_size ~= nil then
          self.batch_size = batch_size
        end

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
        local high = torch.Tensor({self.x_threshold, 100000, self.theta_threshold_radians * 2, 100000})
        self.action_space = rltorch.Discrete(2)
        self.observation_space = rltorch.Box(-high, high)
end
 
function CartPole_v0:step(agent_action)  
        assert(agent_action:size(1) == self.batch_size)
        local rewards = torch.Tensor(self.batch_size)
        local dones = torch.IntTensor(self.batch_size)
        for i = 1, self.batch_size do
            local action = agent_action[i]
            assert(action==1 or action==2, "Invalid action")
            local x=self.state[i][1]
            local x_dot=self.state[i][2]
            local theta=self.state[i][3] 
            local theta_dot = self.state[i][4]
            local reward=0
            
            local force=0
            if (action==2) then force = self.force_mag else force=-self.force_mag end
            local costheta = math.cos(theta)
            local sintheta = math.sin(theta)
            local temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
            local thetaacc = (self.gravity * sintheta - costheta* temp) / (self.length * (4.0/3.0 - self.masspole * costheta * costheta / self.total_mass))
            local xacc  = temp - self.polemass_length * thetaacc * costheta / self.total_mass
            x  = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
            
            self.state[i][1]=x
            self.state[i][2]=x_dot
            self.state[i][3]=theta
            self.state[i][4]=theta_dot
            
            local done =  x < -self.x_threshold 
                    or x > self.x_threshold 
                    or theta < -self.theta_threshold_radians 
                    or theta > self.theta_threshold_radians
            

            if (not done) then
                reward = 1.0
            elseif (self.steps_beyond_done[i]==-1) then
                -- Pole just fell!
                self.steps_beyond_done[i] = 0
                reward = 1.0
            else
                if self.steps_beyond_done[i] == 0 then
                    print("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
                end
                self.steps_beyond_done = self.steps_beyond_done+1
                reward = 0.0
            end
            rewards[i] = reward
            dones[i] = done
        end
          
        return {self.state, rewards, dones} 
end


function CartPole_v0:reset()
    --self.state = torch.rand(4)*0.1-torch.Tensor(4):fill(-0.05)
    self.state = torch.rand(self.batch_size, 4)*0.1
    self.state:add(-0.05)
    self.steps_beyond_done = torch.Tensor(self.batch_size):fill(-1)
   return(self.state)
end 

function CartPole_v0:close()
end

function CartPole_v0:render(arg)
  if (arg.mode=="console") then
    print("Position = "..self.state[1].." / Vitesse = "..self.state[2].." // Angle "..self.state[3].." // Vitesse Angulaire "..self.state[4])
  elseif (arg.mode=="qt") then
      local SX=640
      local SY=480
      local POLE_SIZE=SY/5
          
    
    if (self.__render_widget==nil) then 
      require 'qt'
      require 'qtuiloader'
      require 'qtwidget'
      
      self.__render_widget = qtwidget.newwindow(SX,SY,"CatPole_v0")
    end
    local CART_SX=SX/20
    local CART_SY=SY/20
    self.__render_widget:setcolor("white")
      
    self.__render_widget:showpage()
    self.__render_widget:stroke()
    self.__render_widget:setcolor("red")
    local x=self.state[1]*10
    local VX=x+SX/2
    local VY=SY/2
    self.__render_widget:rectangle(VX-CART_SX/2,VY-CART_SY/2,CART_SX,CART_SY)
    self.__render_widget:fill()
    
    local angle=self.state[3]
    local pole_x=math.sin(angle)*POLE_SIZE
    local pole_y=-math.cos(angle)*POLE_SIZE
    
    self.__render_widget:setlinewidth(5)
    self.__render_widget:setcolor("blue")
    self.__render_widget:moveto(VX,VY)
    self.__render_widget:lineto(VX+pole_x,VY+pole_y)    
    self.__render_widget:stroke()
    self.__render_widget:painter()
    
    sys.sleep(1.0/arg.fps)
  end
end

