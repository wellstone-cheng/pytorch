import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter


class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()#P=0.5
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        self.bn = nn.BatchNorm2d(20)

    def forward(self, x):
        x = F.max_pool2d(self.conv1(x), 2)#self.conv1(x) [13, 10, 24, 24] --> [13, 10, 12, 12]
        x = F.relu(x) + F.relu(-x)  #[13, 10, 12, 12]
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))#x,kernel size;[13, 10, 12, 12]-->[13, 20, 8, 8]-->[13, 20, 4, 4]
        x = self.bn(x) #[13, 20, 4, 4]
        x = x.view(-1, 320) #[13, 20, 4, 4]-->[13,320]
        x = F.relu(self.fc1(x))#[13, 50]
        x = F.dropout(x, training=self.training) #[13, 50]
        x = self.fc2(x #[13, 10]
        x = F.softmax(x, dim=1) #[13, 10]
        return x


dummy_input = torch.rand(13, 1, 28, 28)
#print('dummy_input ',dummy_input)
model = Net1()
print(model)
with SummaryWriter(comment='Net1') as w:
    w.add_graph(model, (dummy_input,))