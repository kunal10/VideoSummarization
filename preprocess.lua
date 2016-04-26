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
cmd:option('-labels','','path to the training labels file')
cmd:option('-cnn_proto', '', 'path to the cnn prototxt')
cmd:option('-cnn_model', '', 'path to the cnn model')
cmd:option('-batch_size', 10, 'batch_size')
cmd:option('-output_size', 1, 'Size of LSTM unit output')
cmd:option('-num_batches', 1000, 'batch_size')
cmd:option('-seq_len', 128, 'seq_len')
cmd:option('-sampling', 10, 'sampling rate. Default 1 in 10')

cmd:option('-out_name', 'data/tvsum50/data_img.h5', 'output name')
cmd:option('-gpuid', 1, 'which gpu to use. -1 = use CPU')
cmd:option('-backend', 'cudnn', 'nn|cudnn')

opt = cmd:parse(arg)
print(opt)

cutorch.setDevice(opt.gpuid)

----------------------------------------------------------------------
-- 1. Load images and labels
print('Train Labels File: ' .. opt.labels)
local labels = io.open(opt.labels, 'r'); 
train_list = {}
test_list = {}
for line in labels:lines() do
  if not line then
    break
  end
  local img_name, score = unpack(line:split(" "));
  local vid = unpack(line:split("_"));
  vid = tonumber(vid)
  print(img_name, score, vid)
  if (vid % 5 == 0) then
    if test_list[vid] == nil then
      print('Adding empty table for test vid id:' .. vid)
      test_list[vid] = {}
    end
    -- Add frame details to this video's table
    table.insert(test_list[vid], {name = img_name, label = tonumber(score)})
    -- print(#test_list)
  else
    if train_list[vid] == nil then
      print('Adding empty table for train vid id:' .. vid)
      train_list[vid] = {}
    end
    -- Add frame details to this video's table
    table.insert(train_list[vid], {name = img_name, label = tonumber(score)})
    -- print(#train_list)
  end
end
labels:close()

local num_train_videos = 0
local num_test_videos = 0
for vid,frames in pairs(train_list) do
  num_train_videos = num_train_videos + 1
  print('Frames for: ' .. vid)
  for fid,frame_info in ipairs(frames) do
    print(fid, frame_info)
  end
end
for vid,frames in pairs(test_list) do
  num_test_videos = num_test_videos + 1
  --print('Frames for: ' .. vid)
  --for fid,frame_info in ipairs(frames) do
  --  print(fid, frame_info)
  --end
end

print('Number of train videos: ' .. num_train_videos);
print('Number of test videos: ' .. num_test_videos);

---------------------------------------------------------------------
-- 2. Load Model
net=loadcaffe.load(opt.cnn_proto, opt.cnn_model, opt.backend);
net:evaluate()
net=net:cuda()

---------------------------------------------------------------------
-- 3. Extract Features
local batch_size = opt.batch_size
local output_size = opt.output_size
local num_batches = opt.num_batches
local seq_len = opt.seq_len
local sampling = opt.sampling

local function printBatch(batch)
  for i,eg in pairs(batch) do
    print('Eg: ' .. i)
    --print('Vid:' .. eg[vid])
    --print('Fid:' .. eg[fid])
  end
end

-- Training Data
local function sampleBatch()
  local sample = {}
  for i = 1,batch_size do
    local vid = 0
    local fid = 0
    -- Randomly select a training video.
    -- while(true) do
      --vid = torch.ceil(torch.uniform() * 40)
      --if (vid % 5 ~= 0) then
      --  break
      --end
    --end
    
    -- For Testing. Remove afterwards
    vid = i+1
    
    -- Randomly select a starting frame such there are sufficient 
    -- frames for seqLen at specified sampling rate.
    local total_frames = #(train_list[vid])
    local last_possible_fid = total_frames - ((seq_len - 1) * sampling)
    fid = torch.ceil(torch.uniform() * last_possible_fid)
    -- Insert vid,fid pair in sample.
    local eg = {['vid'] = vid, ['fid'] = fid}
    print('Adding eg with Vid:' .. eg.vid .. ' and Fid:'.. eg.fid)
    table.insert(sample, {video_id = vid, frame_id = fid})
  end
  printBatch(sample)
  return sample
end

-- Training Data
local inputs, targets = {}, {}
for i = 1,num_batches do
  xlua.progress(i, num_batches)
  inputs[i], targets[i] = {}, {}
  local batch = sampleBatch()
  printBatch(batch)
  print('Processing Batch: ' .. i)
  for j = 1,seq_len do
    -- Process j-th frames for all egs in the batch.
    print('Processing Seq: ' .. j)
    local frames = torch.CudaTensor(batch_size, 3, 224, 224)
    local labels = torch.CudaTensor(batch_size, output_size)
    for k = 1,batch_size do
      -- Read j-th frame from kth eg in the batch.
      print('Processing Video' .. batch[k]['vid'] .. ' Frame: ' .. batch[k]['fid'])
      local cur_vid = batch[k][video_id]
      local cur_fid = batch[k]['fid'] + (seq_len - 1) * sampling
      print(cur_vid, cur_fid)
      print('Loading Frame: ' .. train_list[cur_vid][cur_fid][name])
      frames[k] = loadim(opt.img_dir .. train_list[cur_vid][cur_fid][name]):cuda()
      labels[k] = torch.Tensor(train_list[cur_vid][cur_fid][label]):cuda()
    end
    -- Add features and labels for j-th frames for all egs in the batch.
    net:forward(frames)
    table.insert(inputs[i], net.modules[43].output:clone())
    table.insert(targets[i], labels:clone())
    collectgarbage()
  end
end

print(inputs, targets)
-- Test Data
