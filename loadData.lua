require 'nn'
require 'optim'
require 'torch'
require 'nn'
require 'math'
require 'cunn'
require 'cutorch'
require 'loadcaffe'
require 'image'
require 'hdf5'
cjson=require('cjson') 
require 'xlua'
dofile('loadImage.lua') 

-------------------------------------------------------------------------------
-- Input arguments and options
-------------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Options')
cmd:option('-img_dir','','path to the images directory')
cmd:option('-train_labels','','path to the training labels file')
cmd:option('-test_labels','','path to the test labels file')
cmd:option('-cnn_proto', '', 'path to the cnn prototxt')
cmd:option('-cnn_model', '', 'path to the cnn model')
cmd:option('-batch_size', 2, 'batch_size')

cmd:option('-out_name', 'data/tvsum50/data_img.h5', 'output name')
cmd:option('-gpuid', 1, 'which gpu to use. -1 = use CPU')
cmd:option('-backend', 'cudnn', 'nn|cudnn')

opt = cmd:parse(arg)
print(opt)

cutorch.setDevice(opt.gpuid)
-- open the mdf5 file

----------------------------------------------------------------------
-- 1. Load images and labels
print('Train Labels File: ' .. opt.train_labels)
local train_labels = io.open(opt.train_labels, 'r'); 
local train_list={}
for line in train_labels:lines() do
  if not line then
    break
  end
  print(line)
  local img_name, score = unpack(line:split(" "));
  local img = loadim(opt.img_dir .. 'train/' .. img_name):cuda();
  --table.insert(train_list, { name = img_name, data = img, label = tonumber(score) }); 
  table.insert(train_list, { name = img_name, label = tonumber(score) }); 
end
train_labels:close()
print('Number of train images: ' .. #train_list);

local test_labels = io.open(opt.test_labels, 'r'); 
local test_list={}
for line in test_labels:lines() do
  if not line then
    break
  end
  print(line)
  local img_name, score = unpack(line:split(" "));
  local img = loadim(opt.img_dir .. 'test/' .. img_name):cuda();
  -- table.insert(test_list, { name = img_name, data = img, label = tonumber(score) }); 
  table.insert(test_list, { name = img_name, label = tonumber(score) }); 
end
test_labels:close()
print('Number of test images: ' .. #test_list);

---------------------------------------------------------------------
-- 2. Load Model
net=loadcaffe.load(opt.cnn_proto, opt.cnn_model, opt.backend);
net:evaluate()
net=net:cuda()


---------------------------------------------------------------------
-- 3. Extract Features
local ndims=4096
local batch_size = opt.batch_size
local sz=#train_list
local feat_train=torch.CudaTensor(sz,ndims);
print(string.format('processing %d training images...',sz))
for i=1,sz,batch_size do
    xlua.progress(i, sz)
    r=math.min(sz,i+batch_size-1);
    ims=torch.CudaTensor(r-i+1,3,224,224);
    for j=1,r-i+1 do
        --ims[j]=loadim(train_list[i+j-1]):cuda();
    end
    net:forward(ims)
    feat_train[{{i,r},{}}]=net.modules[43].output:clone();
    print(feat_train:size())
    -- print(feat_train)
    collectgarbage()
end

print('DataLoader loading h5 file: ', 'data_train')
local sz=#test_list
local feat_test=torch.CudaTensor(sz,ndims);
print(string.format('processing %d test images...',sz))
for i=1,sz,batch_size do
    xlua.progress(i, sz)    
    r=math.min(sz,i+batch_size-1);
    ims=torch.CudaTensor(r-i+1,3,224,224);
    for j=1,r-i+1 do
        --ims[j]=loadim(test_list[i+j-1]):cuda();
    end
    net:forward(ims)
    print(feat_test:size())
    -- print(feat_test)
    feat_test[{{i,r},{}}]=net.modules[43].output:clone();
    collectgarbage()
end

local train_h5_file = hdf5.open(opt.out_name, 'w')
train_h5_file:write('/images_train', feat_train:float())
train_h5_file:write('/images_test', feat_test:float())
train_h5_file:close()
