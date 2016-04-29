require 'rnn'
require 'cunn'
require 'cutorch'
require 'xlua'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Options')
cmd:option('-hidden_size', 1000, 'Size of LSTM unit output')
cmd:option('-output_size', 1, 'Size of final output')
cmd:option('-feature_size', 4096, 'Size of input features to LSTM')
cmd:option('-batch_size', 10, 'batch_size')
cmd:option('-num_batches', 200, 'batch_size')
cmd:option('-num_iterations', 10000, 'number of training iterations')

cmd:option('-train_data', 'data/tvsum50/train_data.t7', 'training data file')
cmd:option('-train_targets', 'data/tvsum50/train_data_labels.t7', 'training data labels')
cmd:option('-model_prefix', 'models/blstm', 'model prefix file')
cmd:option('-gpuid', 1, 'which gpu to use. -1 = use CPU')
cmd:option('-backend', 'cudnn', 'nn|cudnn')

opt = cmd:parse(arg)
print(opt)

cutorch.setDevice(opt.gpuid)

-- hyper-parameters 

-- Number of steps to backpropogate gradients.
-- NOTE : LSTM library recommends max value 5.
rho = opt.rho
featureSize = opt.feature_size -- Length of feature vector
hiddenSize = opt.hidden_size
batchSize = opt.batch_size
outputSize = opt.output_size
lr = 0.0001

numBatches = opt.num_batches
numTrainBatches = torch.ceil(0.9 * numBatches)
numIterations = opt.num_iterations

-- forward rnn
-- build simple recurrent neural network
local fwd = nn.FastLSTM(featureSize, hiddenSize)

-- backward rnn (will be applied in reverse order of input sequence)
local bwd = fwd:clone()
bwd:reset() -- reinitializes parameters

-- merges the output of one time-step of fwd and bwd rnns.
-- You could also try nn.AddTable(), nn.Identity(), etc.
local merge = nn.JoinTable(1, 1)  

-- Note that bwd and merge argument are optional and will default to the above.
local brnn = nn.BiSequencer(fwd, bwd, merge)

local rnn = nn.Sequential()
   :add(brnn)
   :add(nn.Sequencer(nn.Linear(hiddenSize*2, outputSize))) -- times two due to JoinTable

--according to http://arxiv.org/abs/1409.2329 this should help model performance 
rnn:getParameters():uniform(-0.1, 0.1)

---- Tip as per https://github.com/Element-Research/rnn/issues/125
rnn:zeroGradParameters()

-- As per comment here: https://github.com/hsheil/rnn-examples/blob/master/part2/main.lua this is essential
rnn:remember('both')

-- print(rnn)

-- build criterion
-- criterion = nn.SequencerCriterion(nn.AbsCriterion())
criterion = nn.SequencerCriterion(nn.SmoothL1Criterion())

-- Load inputs and targets
inputs = torch.load(opt.train_data)
targets = torch.load(opt.train_targets)

if (opt.gpuid > 0) then
  criterion = criterion:cuda()
  rnn = rnn:cuda()
end

local function printDebugInfo(output, target)
    print('\nPredictions:')
    for i,j in ipairs(output) do
        print(i, output[i], target[i])
    end
end

rnn:training()
-- Iterate over all input batches and learn params.
for i = 1,numIterations do
    xlua.progress(i, numIterations)
    local index = (i % numTrainBatches)
    
    if (index == 0) then 
      index = numTrainBatches
    end

    local outputs = rnn:forward(inputs[index])
    -- printDebugInfo(outputs, targets[i])
    
    local err = criterion:forward(outputs, targets[index])
    print(string.format("Iteration %d ; err = %f ", i, err))

    -- 3. backward sequence through rnn (i.e. backprop through time)
    local gradOutputs = criterion:backward(outputs, targets[index])
    local gradInputs = rnn:backward(inputs[index], gradOutputs)

    -- 4. update
    rnn:updateParameters(lr)
    rnn:zeroGradParameters()
    rnn:forget()

    -- Evaluate model on validation set 
    if (i % 100 == 0) then
      rnn:evaluate()
      local valerr = 0 
      local iters = (numBatches - numTrainBatches)
      for j = (numTrainBatches + 1),numBatches do
        local predicted = rnn:forward(inputs[j])
        local batcherr = criterion:forward(predicted, targets[j])
        valerr = valerr + batcherr
        -- print(string.format("Batch %d ; Val err = %f ", j, batcherr)) 
      end
      print(string.format("Iteration %d ; Avg Val Err = %f\n", i, valerr / iters)) 
      -- Turn on training mode again
      rnn:training()
    end

    if (i % 500 == 0) then
      lr = lr * 0.8
    end

    if (i == 500) then
      torch.save(opt.model_prefix .. '500' .. '.t7', rnn)
    end
end

print('Saving Trained Model')
torch.save(opt.model_prefix .. numIterations .. '.t7', rnn)
