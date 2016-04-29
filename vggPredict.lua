require 'rnn'
require 'cunn'
require 'cutorch'
require 'xlua'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Options')

cmd:option('-seq_len', 128, 'Sequence length in training data')
cmd:option('-test_data', 'data/tvsum50/test_data.t7', 'test data file')
cmd:option('-test_targets', 'data/tvsum50/test_data_labels.t7', 'test data labels')
cmd:option('-model', 'models/vgg10000.t7', 'model to be evaluated')
cmd:option('-output_file', 'result/vgg_predictions10000.txt', 'model predictions')
cmd:option('-gpuid', 1, 'which gpu to use. -1 = use CPU')
cmd:option('-backend', 'cudnn', 'nn|cudnn')

opt = cmd:parse(arg)
print(opt)

cutorch.setDevice(opt.gpuid)

seqLen = opt.seq_len
-- Open file for writing outputs.
file = io.open(opt.output_file, "w")

local net = torch.load(opt.model)

print(net)

-- build criterion
-- criterion = nn.SequencerCriterion(nn.AbsCriterion())
criterion = nn.SmoothL1Criterion()

-- Load inputs and targets
inputs = torch.load(opt.test_data)
targets = torch.load(opt.test_targets)
print(targets[i])
-- Convert to cuda compatible versions
if (opt.gpuid > 0) then
  criterion = criterion:cuda()
  net = net:cuda()
end

local function writeToFile(predicted, target)
end

-- Iterate over all test data
local testSize = #inputs
for i = 1,testSize do
    xlua.progress(i, testSize)
    local err = 0
    for k = 1,#(inputs[i]) do
        local predicted = net:forward(inputs[i][k])
        local batcherr = criterion:forward(predicted, targets[i][k])
        err = err + batcherr
        writeToFile(predicted, targets[i][k])
        if (k == #(inputs[i])) then
    	    file:write(string.format("%f,", predicted[1][1]))
        else
    	    file:write(string.format("%f", predicted[1][1]))
        end 
	print(string.format("Test Eg: %d Seq No: %d ; err = %f, avg err = %f ", i, k, err, err/(#targets[i])))
    end  
    file:write("\n")
end

-- Close the output file
print("Finised writing output. Closing the output file")
file:close()
