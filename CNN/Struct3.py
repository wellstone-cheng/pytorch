import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter

class Net3(torch.nn.Module):
    def __init__(self):
        super(Net3, self).__init__()
        self.conv=torch.nn.Sequential()
        self.conv.add_module("conv1",torch.nn.Conv2d(3, 32, 3, 1, 1))
        self.conv.add_module("relu1",torch.nn.ReLU())
        self.conv.add_module("pool1",torch.nn.MaxPool2d(2))
        self.dense = torch.nn.Sequential()
        self.dense.add_module("dense1",torch.nn.Linear(32 * 3 * 3, 128))
        self.dense.add_module("relu2",torch.nn.ReLU())
        self.dense.add_module("dense2",torch.nn.Linear(128, 10))

    def forward(self, x):
        conv_out = self.conv(x)
        res = conv_out.view(conv_out.size(0), -1)
        out = self.dense(res)
        return out

print("Method 3:")
model3 = Net3()
print(model3)
dummy_input = torch.rand(2, 3, 6, 6)
with SummaryWriter(comment='Net2') as w:
    w.add_graph(model3, (dummy_input,))