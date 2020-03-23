import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
class Net1(torch.nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 3, 1, 1)
        self.dense1 = torch.nn.Linear(32 * 3 * 3, 128)
        self.dense2 = torch.nn.Linear(128, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)#[2, 32, 6, 6]-->[2, 32, 3, 3]
        x = x.view(x.size(0), -1)#x.size(0) =2 ; [2, 32, 3, 3]-->[2, 288]
        x = F.relu(self.dense1(x))#[2, 288]-->[2, 128]
        x = self.dense2(x)#[2, 128]-->[2, 10]
        return x
print("Method 1:")
model1 = Net1()
print(model1)
dummy_input = torch.rand(2, 3, 6, 6)
with SummaryWriter(comment='Net1') as w:
    w.add_graph(model1, (dummy_input,))