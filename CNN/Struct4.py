import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from collections import OrderedDict

class Net4(torch.nn.Module):
    def __init__(self):
        super(Net4, self).__init__()
        self.conv = torch.nn.Sequential(
            OrderedDict(
                [
                    ("conv1", torch.nn.Conv2d(3, 32, 3, 1, 1)),
                    ("relu1", torch.nn.ReLU()),
                    ("pool", torch.nn.MaxPool2d(2))
                ]
            ))

        self.dense = torch.nn.Sequential(
            OrderedDict([
                ("dense1", torch.nn.Linear(32 * 3 * 3, 128)),
                ("relu2", torch.nn.ReLU()),
                ("dense2", torch.nn.Linear(128, 10))
            ])
        )

    def forward(self, x):
        conv_out = self.conv(x)
        res = conv_out.view(conv_out.size(0), -1)
        out = self.dense(res)
        return out

print("Method 4:")
model4 = Net4()
print(model4)
dummy_input = torch.rand(2, 3, 6, 6)
with SummaryWriter(comment='Net2') as w:
    w.add_graph(model4, (dummy_input,))