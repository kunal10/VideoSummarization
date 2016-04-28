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
cmd:option('-img_dir','data/tvsum50/frames_all/','path to the images directory')
cmd:option('-labels','data/tvsum50/frame_file_all.txt','path to the labels file')
cmd:option('-cnn_proto', 'models/VGG_ILSVRC_19_layers_deploy.prototxt', 'path to the cnn prototxt')
cmd:option('-cnn_model', 'models/VGG_ILSVRC_19_layers.caffemodel', 'path to the cnn model')
cmd:option('-batch_size', 10, 'batch_size')
cmd:option('-output_size', 1, 'Size of LSTM unit output')
cmd:option('-num_batches', 200, 'number of batches')
cmd:option('-seq_len', 128, 'seq_len')
cmd:option('-sampling', 10, 'sampling rate. Default 1 in 10')

cmd:option('-out_train_data', 'data/tvsum50/train_data.t7', 'training data file')
cmd:option('-out_train_labels', 'data/tvsum50/train_data_labels.t7', 'training data labels')
cmd:option('-out_test_data', 'data/tvsum50/test_data.t7', 'test data file')
cmd:option('-out_test_labels', 'data/tvsum50/test_data_labels.t7', 'test data labels')
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
    print('Vid:' .. eg.video_id)
    print('Fid:' .. eg.frame_id)
  end
end

-- Training Data
local function sampleBatch()
  local sample = {}
  for i = 1,batch_size do
    local vid = 0
    local fid = 0
    -- Randomly select a training video.
    while(true) do
      vid = torch.ceil(torch.uniform() * 40)
      if (vid % 5 ~= 0) then
        break
      end
    end
    -- Randomly select a starting frame such there are sufficient 
    -- frames for seqLen at specified sampling rate.
    local total_frames = #(train_list[vid])
    local last_possible_fid = total_frames - ((seq_len - 1) * sampling)
    -- print('TotalFrames, LastPossibleFid : ' .. total_frames, last_possible_fid)
    assert(last_possible_fid > 0)
    fid = torch.ceil(torch.uniform() * last_possible_fid)
    -- Insert vid,fid pair in sample.
    table.insert(sample, {video_id = vid, frame_id = fid})
  end
  -- printBatch(sample)
  return sample
end

-- Training Data
local train_inputs, train_targets = {}, {}
for i = 1,num_batches do
  xlua.progress(i, num_batches)
  train_inputs[i], train_targets[i] = {}, {}
  local batch = sampleBatch()
  print('Processing Batch: ' .. i)
  for j = 1,seq_len do
    -- Process j-th frames for all egs in the batch.
    -- print('Processing Seq: ' .. j)
    local frames = torch.CudaTensor(batch_size, 3, 224, 224)
    local labels = torch.CudaTensor(batch_size, output_size)
    for k = 1,batch_size do
      -- Read j-th frame from kth eg in the batch.
      local cur_vid = batch[k].video_id
      local cur_fid = batch[k].frame_id + (j - 1) * sampling
      -- print(cur_vid, cur_fid)
      -- print('Loading Frame: ' .. train_list[cur_vid][cur_fid].name)
      frames[k] = loadim(opt.img_dir .. train_list[cur_vid][cur_fid].name):cuda()
      labels[k] = torch.CudaTensor({train_list[cur_vid][cur_fid].label})
    end
    -- Add features and labels for j-th frames for all egs in the batch.
    net:forward(frames)
    table.insert(train_inputs[i], net.modules[43].output:clone())
    table.insert(train_targets[i], labels:clone())
    collectgarbage()
  end
  -- Save partial dataset
  if (i % 50 == 0) then
    torch.save(opt.out_train_data .. i, train_inputs)
    torch.save(opt.out_train_labels .. i, train_targets)
  end
end

-- Test Data
local test_inputs, test_targets = {}, {}
-- Need to keep batch size 1 since all test videos can have different lengths. 
batch_size = 1
for i = 1,num_test_videos do
  xlua.progress(i, num_test_videos)
  test_inputs[i], test_targets[i] = {}, {}
  print('Processing Test Video: ' .. i)
  local vid = i * 5
  local fid = torch.ceil(torch.uniform() * sampling)
  while fid < #(test_list[vid]) do
    -- Process j-th frame for test video.
    local frames = torch.CudaTensor(batch_size, 3, 224, 224)
    local labels = torch.CudaTensor(batch_size, output_size)
    for k = 1,batch_size do
      frames[k] = loadim(opt.img_dir .. test_list[vid][fid].name):cuda()
      labels[k] = torch.CudaTensor({test_list[vid][fid].label})
    end
    -- Add features and labels for j-th frames for all egs in the batch.
    net:forward(frames)
    table.insert(test_inputs[i], net.modules[43].output:clone())
    table.insert(test_targets[i], labels:clone())
    collectgarbage()
    -- Increment fid
    fid = fid + sampling
  end
end

torch.save(opt.out_test_data, test_inputs)
torch.save(opt.out_test_labels, test_targets)
