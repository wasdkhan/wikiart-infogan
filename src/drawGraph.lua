require('torch')    -- Essential Torch utilities
require('nn')       -- Neural network building blocks
require('cutorch')  -- 'torch' on the GPU
require('cunn')     -- 'nn' on the GPU
require('nngraph')
local pdist = require('pdist')

local wikiartmodel_builder = require('wikiartmodel_builder')

local n_gen_inputs = 104
local n_salient_vars = 42
local n_noise_vars = 104 - 42
local batch_size = 128

local dist = pdist.Hybrid()
  :add(pdist.Categorical{n = 10, probs = torch.CudaTensor(10):fill(1 / 10)})
  :add(pdist.Categorical{n = 10, probs = torch.CudaTensor(10):fill(1 / 10)})
  :add(pdist.Categorical{n = 10, probs = torch.CudaTensor(10):fill(1 / 10)})
  :add(pdist.Categorical{n = 10, probs = torch.CudaTensor(10):fill(1 / 10)})
  :add(pdist.Gaussian{
    n = n_salient_vars - 40,
    mean = torch.CudaTensor(n_salient_vars - 40):fill(0),
    stddev = torch.CudaTensor(n_salient_vars - 40):fill(1),
    fixed_stddev = true
  })

generator, discriminator_body, discriminator_head, info_head =
  wikiartmodel_builder.build_infogan(n_gen_inputs, dist:n_params())

discriminator = nn.Sequential()
  :add(discriminator_body)
  :add(nn.ConcatTable()
    :add(discriminator_head)
    :add(info_head)
  )

generator:cuda()
discriminator:cuda()

gen_input = torch.CudaTensor()
gen_input:resize(batch_size, n_gen_inputs)
dist:sample(gen_input:narrow(2, 1, n_salient_vars), dist.prior_params)
gen_input:narrow(2, n_salient_vars + 1, n_noise_vars):normal(0, 1)
--generator:forward(gen_input)

generateGraph = require 'optnet.graphgen'

-- visual properties of the generated graph
-- follows graphviz attributes
graphOpts = {
displayProps =  {shape='ellipse',fontsize=14, style='solid'},
nodeData = function(oldData, tensor)
  return oldData .. '\n' .. 'Size: '.. tensor:numel()
end
}

--g = generateGraph(generator, gen_input, graphOpts)
--graph.dot(g,'wikiart64-g', 'wikiart64-g')

--output = generator:forward(gen_input)
--g = generateGraph(discriminator, output, graphOpts)
--graph.dot(g,'wikiart64-d', 'wikiart64-d')
