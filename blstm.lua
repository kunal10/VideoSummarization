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
cmd:option('-num_batches', 1000, 'batch_size')

cmd:option('-train_data', 'data/tvsum50/train_data.t7', 'training data file')
cmd:option('-train_targets', 'data/tvsum50/train_data_labels.t7', 'training data labels')
cmd:option('-out_model_prefix', 'models/blstm', 'model prefix file')
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
for i = 1,numBatches do
    xlua.progress(i, numBatches)
    local outputs = rnn:forward(inputs[i])
    printDebugInfo(outputs, targets[i])
    
    local err = criterion:forward(outputs, targets[i])
    print(string.format("Iteration %d ; MSE err = %f ", i, err))

    -- 3. backward sequence through rnn (i.e. backprop through time)
    local gradOutputs = criterion:backward(outputs, targets[i])
    local gradInputs = rnn:backward(inputs[i], gradOutputs)

    -- 4. update
    rnn:updateParameters(lr)
    rnn:zeroGradParameters()
    rnn:forget()

    --if (i % 10 == 0) then
    --  lr = lr * 0.5
    --end
end
