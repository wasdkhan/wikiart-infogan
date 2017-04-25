require 'image'
require 'sys'

cmd = torch.CmdLine()
cmd:text('Options:')
cmd:option('-dir', "apple64", 'directory name')
cmd:option('-out', "apple64-train.t7", 'output file name')
cmd:option('-cfile', "train.txt", 'name of class file inside directory')
cmd:text()
opt = cmd:parse(arg or {})

local dir = opt.dir
if dir:sub(dir:len(), dir:len()) ~= '/' then
  dir = dir..'/'
end
local outputFile = opt.out
local classFile = dir..(opt.cfile)

local f = io.open(classFile, "r")
local fileOk = false
if f ~= nil then
  io.close(f)
  fileOk = true
end

if not fileOk then
  print('[error]class file not found, rename to match dir.txt, or dir not found, use -dir to specify`')
else
  local sampleSize = tonumber(sys.execute('wc -l <'..classFile))
  print("total lines: "..sampleSize)

  --local data = torch.ByteTensor(sampleSize, 3, 32, 32)
  local data = torch.ByteTensor(sampleSize, 3, 256, 256)
  local label = torch.ByteTensor(sampleSize)
  local i = 1

  for line in io.lines(classFile) do
    local name, class = line:match("(%S+) ([0-9]+)")
    name = dir..name
    local ok, img = pcall(image.load, name, 3, 'byte')
    if not ok then
      print('[error]'..img..'`')
    else
      label[i] = tonumber(class) 
      data[i] = img
      i = i + 1
    end
  end

  local out = {}
  out.data = data
  out.label = label
  print(out)
  torch.save(outputFile, out)
end
