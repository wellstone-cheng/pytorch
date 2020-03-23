import torch

x = torch.randn(10, 20)  # 输入的维度是（10，20）,有10个１维数据，每个１维数据中有20个数据:10*20

m = torch.nn.Linear(20, 30)  #权重矩阵m.weight 有30个１维数据，每个１维数据中有20个数据
output = m(x)
print('m.weight.shape:\n ', m.weight.shape)#权重矩阵为30*20 --> 专置权重矩阵维20*30;.shape表示数据的尺寸
print('m.bias.shape:\n', m.bias.shape)#偏置
print('output.shape:\n', output.shape)

ans = torch.mm(x, m.weight.t()) + m.bias
print('ans.shape:\n', ans.shape)
