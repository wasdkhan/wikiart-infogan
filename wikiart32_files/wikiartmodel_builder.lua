require('cudnn')
local nninit = require('nninit')

local model_builder = {}

local Seq = nn.Sequential
local ReLU = cudnn.ReLU

local function SpatBatchNorm(n_outputs)
  return nn.SpatialBatchNormalization(n_outputs, 1e-5, 0.1)
    :init('weight', nninit.normal, 1.0, 0.02) -- Gamma
    :init('bias', nninit.constant, 0)         -- Beta
end

local function BatchNorm(n_outputs)
  return nn.BatchNormalization(n_outputs, 1e-5, 0.1)
    :init('weight', nninit.normal, 1.0, 0.02) -- Gamma
    :init('bias', nninit.constant, 0)         -- Beta
end

local function Conv(...)
  local conv = cudnn.SpatialConvolution(...)
    :init('weight', nninit.normal, 0.0, 0.02)
    :init('bias', nninit.constant, 0)

  -- Use deterministic algorithms for convolution
  conv:setMode(
    'CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM',
    'CUDNN_CONVOLUTION_BWD_DATA_ALGO_1',
    'CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1')

  return conv
end

local function FullConv(...)
  local conv = cudnn.SpatialFullConvolution(...)
    :init('weight', nninit.normal, 0.0, 0.02)
    :init('bias', nninit.constant, 0)

  -- Use deterministic algorithms for convolution
  conv:setMode(
    'CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM',
    'CUDNN_CONVOLUTION_BWD_DATA_ALGO_1',
    'CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1')

  return conv
end

local function LeakyReLU(leakiness, in_place)
  leakiness = leakiness or 0.01
  in_place = in_place == nil and true or in_place
  return nn.LeakyReLU(leakiness, in_place)
end

local function Linear(...)
  return nn.Linear(...)
    :init('weight', nninit.normal, 0.0, 0.02)
    :init('bias', nninit.constant, 0)
end

function model_builder.build_infogan(n_gen_inputs, n_salient_params)
  local generator = Seq()
    -- n_gen_inputs
    :add(Linear(n_gen_inputs, 512*3*3))
    :add(BatchNorm(512*3*3))
    :add(ReLU(true))
    :add(nn.Reshape(512, 3, 3))
    -- 512 x 3 x 3
    :add(FullConv(512, 256, 3,3, 2,2))
    :add(SpatBatchNorm(256))
    :add(ReLU(true))
    -- 256 x 7 x 7
    :add(FullConv(256, 128, 4,4, 2,2, 1,1))
    :add(SpatBatchNorm(128))
    :add(ReLU(true))
    -- 128 x 14 x 14
    :add(FullConv(128, 3, 4,4, 2,2, 1,1))
    :add(nn.Tanh())
    -- 3 x 28 x 28

  local discriminator_body = Seq()
    -- 3 x 28 x 28
    :add(Conv(3, 128, 4,4, 2,2, 1,1))
    :add(LeakyReLU())
    -- 128 x 14 x 14
    :add(Conv(128, 256, 4,4, 2,2, 1,1))
    :add(SpatBatchNorm(256))
    :add(LeakyReLU())
    -- 256 x 7 x 7
    :add(Conv(256, 512, 3,3, 2,2))
    :add(SpatBatchNorm(512))
    :add(LeakyReLU())
    -- 512 x 3 x 3
    :add(Conv(512, 756, 3, 3))
    :add(nn.Sigmoid())
    :add(nn.Reshape(756))
    -- 756

  local discriminator_head = Seq()
    -- 756
    :add(Linear(756, 1))
    :add(nn.Sigmoid())
    -- 1

  local info_head = Seq()
    -- 756
    :add(Linear(756, 95))
    :add(BatchNorm(95))
    :add(LeakyReLU())
    -- 95
    :add(Linear(95, n_salient_params))
    -- n_salient_params

  return generator, discriminator_body, discriminator_head, info_head
end

return model_builder
