-- Data loader for svhn

local tnt = require('torchnet')
local argcheck = require('argcheck')

local SvhnDataset = torch.class('SvhnDataset', {})

SvhnDataset.__init = argcheck{
  {name = 'self', type = 'SvhnDataset'},
  {name = 'data_file', type = 'string'},
  call = function(self, data_file)
    local raw_data = torch.load(data_file, 'ascii')
    self.inputs = raw_data.X:transpose(3,4):float():div(255)
    self.targets = raw_data.y[1]
  end
}

SvhnDataset.make_iterator = argcheck{
  {name = 'self', type = 'SvhnDataset'},
  {name = 'batch_size', type = 'number', default = 32},
  {name = 'n_threads', type = 'number', default = 8},
  call = function(self, batch_size, n_threads)
    local inputs = self.inputs
    local targets = self.targets

    local function load_example_from_index(index)
      return {
        input = inputs[index],
        target = targets:narrow(1, index, 1)
      }
    end

    local gen = torch.Generator()
    torch.manualSeed(gen, 1234)
    local indices = torch.randperm(gen, inputs:size(1)):long()

    return tnt.ParallelDatasetIterator{
      ordered = true,
      nthread = n_threads,
      closure = function()
        local tnt = require('torchnet')

        return tnt.BatchDataset{
          batchsize = batch_size,
          dataset = tnt.ListDataset{
            list = indices,
            load = load_example_from_index
          }
        }
      end
    }
  end
}

return SvhnDataset
