 require('paths')
 --- This class implements a set of tools for reading files

local RLFile = torch.class('rltorch.RLFile'); 

--- Read a libsvm file and returns a matrix of data and a matrix of labels. Labels are reindexed
function RLFile:read_libsvm(filename)
  local data = svm.ascread('datasets/breast-cancer_scale')  
  local nb_examples=#data
  local index_categories={}
  local nb_cat=0
  local max_feature_index=0
  for i=1,nb_examples do    
    local vf,mf=data[i][2][1]:max(1)
    if (vf[1]>max_feature_index) then max_feature_index=vf[1] end
    
    local cat=data[i][1]
    if (index_categories[cat]==nil) then
      index_categories[cat]=nb_cat+1 
      nb_cat=nb_cat+1
    end    
  end
  
  local examples=torch.Tensor(nb_examples,max_feature_index):fill(0)
  local labels=torch.Tensor(nb_examples,1):fill(0)
  
  for i=1,nb_examples do
    local i_f=data[i][2][1]
    local v_f=data[i][2][2]
    for k=1,i_f:size(1) do examples[i][i_f[k]]=v_f[k] end
    labels[i]=index_categories[data[i][1]]      
  end
  return {examples,labels,#index_categories}   
end

--- generate a split (train and test) over a data and labels matrices. 
function RLFile:split_train_test(data,labels,proportion_train)
 local nb_examples=data:size(1)
  local train_or_test={}
  local nb_train=0
  local nb_test=0
  for i=1,nb_examples do
    if (math.random()<proportion_train) then train_or_test[i]="train" nb_train=nb_train+1 else train_or_test[i]="test"; nb_test=nb_test+1 end  
  end
  
  local training_examples=torch.Tensor(nb_train,data:size(2)):fill(0)
  local training_labels=torch.Tensor(nb_train,1):fill(0)
  local testing_examples=torch.Tensor(nb_test,data:size(2)):fill(0)
  local testing_labels=torch.Tensor(nb_test,1):fill(0)
  
  local pos_train=1
  local pos_test=1
  for i=1,nb_examples do
    if (train_or_test[i]=="train") then
      training_examples[pos_train]:copy(data[i])
      training_labels[pos_train]:copy(labels[i])
      pos_train=pos_train+1
    else
      testing_examples[pos_test]:copy(data[i])
      testing_labels[pos_test]:copy(labels[i])
      pos_test=pos_test+1
    end
  end
  return {training_examples,training_labels,testing_examples,testing_labels}   
end

---- Read a set of libsvm files and returns a table of examples + table of labels. 
-- if sample_rate~=nil, then sample_rate is the percentage of files kept
function RLFile:read_multiple_libsvm(directory,suffix,sample_rate)
  local files=paths.dir(directory)
  
  local data={}
  local labels={}
  local pos=1
  
  for k,v in ipairs(files) do   
    if(paths.filep('directory'..'/'..v)) then
      if (string.sub(v,-len(suffix))==suffix) then
        print("=== Reading "..directory.."/"..v)
        if (sample_rate==nil) then
          local e,l=self:read_libsvm("directory".."/"..v)
          data[pos]=e
          labels[pos]=l
          pos=pos+1
        elseif (math.random()<sample_rate) then
          local e,l=self:read_libsvm("directory".."/"..v)
          data[pos]=e
          labels[pos]=l
          pos=pos+1
        else
          print("=== File "..directory.."/"..v.." skipped.")
        end
      end
      collectgarbage()
    end    
  end
  print("==== "..(pos-1).." files.")
  return{data,labels}  
end


