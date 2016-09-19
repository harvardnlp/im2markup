require 'nn'

function createCriterion(output_size)
    local weights = torch.ones(output_size)
    weights[1] = 0
    --for i = 4, 53+3 do
    --    weights[i] = 16
    --end
    criterion = nn.ClassNLLCriterion(weights)
    criterion.sizeAverage = false
    return criterion
end
