require 'nn'
require 'cunn'
require 'cutorch'
require 'xlua'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Options')
cmd:option('-seq_len', 128, 'Sequence length in training data')
cmd:option('-feature_size', 4096, 'Size of VGG features used for training')
cmd:option('-output_size', 1, 'Size of final output')
cmd:option('-batch_size', 10, 'Batch size')
cmd:option('-num_batches', 200, 'batch_size')
cmd:option('-num_iterations', 10000, 'number of training iterations')

cmd:option('-train_data', 'data/tvsum50/train_data200.t7', 'training data file')
cmd:option('-train_targets', 'data/tvsum50/train_data_labels200.t7', 'training data labels')
cmd:option('-model_prefix', 'models/vgg', 'model prefix file')
cmd:option('-gpuid', 1, 'which gpu to use. -1 = use CPU')
cmd:option('-backend', 'cudnn', 'nn|cudnn')

opt = cmd:parse(arg)
print(opt)

cutorch.setDevice(opt.gpuid)

-- hyper-parameters 
featureSize = opt.feature_size
outputSize = opt.output_size
seqLen = opt.seq_len
lr = 0.00001

batchSize = opt.batch_size
seqLen = opt.seq_len 
numBatches = opt.num_batches
numTrainBatches = torch.ceil(0.9 * numBatches)
numIterations = opt.num_iterations

local net = nn.Sequential():add(nn.Linear(featureSize, outputSize))
net:getParameters():uniform(-0.1, 0.1)
net:zeroGradParameters()
print(net)

-- build criterion
-- criterion = nn.Criterion(nn.AbsCriterion())
criterion = nn.SmoothL1Criterion()

-- Load inputs and targets
inputs = torch.load(opt.train_data)
targets = torch.load(opt.train_targets)

if (opt.gpuid > 0) then
  criterion = criterion:cuda()
  net = net:cuda()
end

local function printDebugInfo(output, target)
    print('\nPredictions:')
    for i,j in ipairs(output) do
        print(i, output[i], target[i])
    end
end

net:training()
-- Iterate over all input batches and learn params.
local batchIndex = 0
for i = 1,numIterations do
    xlua.progress(i, numIterations)
    
    local seqNo = i % seqLen
    if (seqNo == 0) then
      seqNo = seqLen
    end

    if (i % seqLen == 1) then 
      batchIndex = batchIndex + 1
      if (batchIndex == (numTrainBatches + 1)) then
        batchIndex = 1
      end
    end

    -- print("\nBatchIndex: " .. batchIndex .. " SeqNo: " ..  seqNo)
    local outputs = net:forward(inputs[batchIndex][seqNo])
    local err = criterion:forward(outputs, targets[batchIndex][seqNo])
    print(string.format("Iteration %d ; err = %f ", i, err))

    -- 3. backward sequence through net (i.e. backprop through time)
    local gradOutputs = criterion:backward(outputs, targets[batchIndex][seqNo])
    local gradInputs = net:backward(inputs[batchIndex][seqNo], gradOutputs)

    -- 4. update
    net:updateParameters(lr)
    net:zeroGradParameters()

    -- Evaluate model on validation set 
    if (i % 100 == 0) then
       net:evaluate()
       local valerr = 0 
       local iters = (numBatches - numTrainBatches)
       for j = (numTrainBatches + 1),numBatches do
         for k = 1,seqLen do
           local predicted = net:forward(inputs[j][k])
           local batcherr = criterion:forward(predicted, targets[j][k])
           valerr = valerr + batcherr
           -- print(string.format("Batch %d ; Val err = %f ", j, batcherr)) 
         end
       end
      print(string.format("TotalValErr = %f, SeqLen = %f, BatchSize = %f", valerr, seqLen, batchSize)) 
      print(string.format("Avg Val Err = %f\n", valerr / (iters * seqLen))) 
      -- Turn on training mode again
      net:training()
    end

    if (i % 500 == 0) then
      lr = lr * 0.8
    end

    if (i == 500) then
      torch.save(opt.model_prefix .. '500' .. '.t7', net)
    end
end

print('Saving Trained Model')
torch.save(opt.model_prefix .. numIterations .. '.t7', net)
