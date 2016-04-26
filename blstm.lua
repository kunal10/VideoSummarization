require 'rnn'

-- hyper-parameters 
rho = 5 -- sequence length
featureSize = 2 -- Length of feature vector
hiddenSize = 7
batchSize = 3
outputSize = 1
lr = 0.01

-- forward rnn
-- build simple recurrent neural network
local fwd = nn.LSTM(
   featureSize, hiddenSize, rho
)

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
-- rnn:getParameters():uniform(-0.1, 0.1)
-- print(rnn)

-- build criterion
criterion = nn.SequencerCriterion(nn.MSECriterion())

-- build dummy dataset
numBatches = 2
seqLength = 5
inputs, targets = {}, {}

for i = 1,numBatches do
    inputs[i], targets[i] = {}, {}
    for j = 1,seqLength do
        table.insert(inputs[i], torch.randn(batchSize,featureSize))
        table.insert(targets[i], torch.randn(batchSize,outputSize))
    end
end

-- print(inputs, targets)
-- print(rnn:forward(inputs[1]))
-- Iterate over all input batches and learn params.

local function printDebugInfo(input, output, target)
    print('Printing Inputs:')
    for i,j in ipairs(input) do
        print(i, input[i])
    end
    print('Printing Outputs:')
    for i,j in ipairs(output) do
        print(i, output[i])
    end
    print('Printing Targets:')
    for i,j in ipairs(target) do
        print(i, target[i])
    end
end

for i = 1,numBatches do
    local outputs = rnn:forward(inputs[i])
    printDebugInfo(inputs[i], outputs, targets[i])
    
    rnn:zeroGradParameters()
    local err = criterion:forward(outputs, targets[i])
    print(string.format("Iteration %d ; MSE err = %f ", i, err))

    -- 3. backward sequence through rnn (i.e. backprop through time)
    local gradOutputs = criterion:backward(outputs, targets[i])
    local gradInputs = rnn:backward(inputs[i], gradOutputs)

    -- 4. update
    rnn:updateParameters(lr)
    rnn:forget()
end
