import torch
import torch.nn as nn
import torch.autograd as autograd

m = nn.Dropout(p=0.5)
n = nn.Dropout2d(p=0.5)
input = autograd.Variable(torch.randn(2, 6, 3)) ## 对dim=1维进行随机置为0
print(input)
print('****************************************************')
print(m(input))
print('****************************************************')
print(n(input))
