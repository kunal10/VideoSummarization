require 'rnn'
require 'cunn'
require 'cutorch'
require 'xlua'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Options')

cmd:option('-test_data', 'data/tvsum50/test_data.t7', 'test data file')
cmd:option('-test_targets', 'data/tvsum50/test_data_labels.t7', 'test data labels')
cmd:option('-model', 'models/vgg10000.t7', 'model to be evaluated')
cmd:option('-output_file', 'result/vgg_predictions10000.txt', 'model predictions')
cmd:option('-gpuid', 1, 'which gpu to use. -1 = use CPU')
cmd:option('-backend', 'cudnn', 'nn|cudnn')

opt = cmd:parse(arg)
print(opt)

cutorch.setDevice(opt.gpuid)

-- Open file for writing outputs.
file = io.open(opt.output_file, "w")

local net = torch.load(opt.model)

print(net)

-- build criterion
-- criterion = nn.SequencerCriterion(nn.AbsCriterion())
criterion = nn.SequencerCriterion(nn.SmoothL1Criterion())

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
    for i,j in ipairs(predicted) do
	if (i == #predicted) then
          file:write(string.format("%f", predicted[i][1][1]))
        else
          file:write(string.format("%f,", predicted[i][1][1]))
        end
    end
    file:write("\n")
end

-- Iterate over all test data
local testSize = #inputs
for i = 1,testSize do
    xlua.progress(i, testSize)
    
    local predicted = net:forward(inputs[i])
    writeToFile(predicted, targets[i])
    -- printDebugInfo(predicted, targets[i])
    
    local err = criterion:forward(predicted, targets[i])
    print(string.format("Test Eg: %d Seq Len: %d ; err = %f, avg err = %f ", i, #targets[i], err, err/(#targets[i])))
end

-- Close the output file
print("Finised writing output. Closing the output file")
file:close()
