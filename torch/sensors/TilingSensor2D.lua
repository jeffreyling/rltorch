 
 
local TilingSensor2D = torch.class('rltorch.TilingSensor2D','rltorch.Sensor'); 

--- Discretize each input dimension considering a particular number of buckets. It works for 2D observations (e.g MountainCar)
function TilingSensor2D:__init(observation_space,nb_buckets_dim_1,nb_buckets_dim_2)
  rltorch.Sensor.__init(self,observation_space)
  self.n1=nb_buckets_dim_1
  self.n2=nb_buckets_dim_2
  self.p=torch.Tensor(1,self.n1*self.n2):fill(0)
end

function TilingSensor2D:process(observation)    
  self.p:fill(0)
  local v=observation:clone()-self.observation_space.low 
  v:cdiv(self.observation_space.high)  
  local i1=math.floor(v[1]*self.n1)
  local i2=math.floor(v[2]*self.n2)
  if (i1>=self.n1) then i1=self.n1-1 end
  if (i2>=self.n2) then i2=self.n2-1 end
  --print(i1.." et "..i2)
  local i=i1*self.n2+i2
  i=i+1
  self.p[1][i]=1
  return self.p
end

function TilingSensor2D:size()
  return(self.n1*self.n2)
end


