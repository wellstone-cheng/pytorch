import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter

class Net2(torch.nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2))
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(32 * 3 * 3, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 10)
        )

    def forward(self, x):
        conv_out = self.conv1(x)
        res = conv_out.view(conv_out.size(0), -1)
        out = self.dense(res)
        return out

print("Method 2:")
model2 = Net2()
print(model2)
dummy_input = torch.rand(2, 3, 6, 6)
with SummaryWriter(comment='Net2') as w:
    w.add_graph(model2, (dummy_input,))