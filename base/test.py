import torch
x=[1,2,3,4]
x_tenser=torch.Tensor(x)
x_tenser_0=torch.unsqueeze(x_tenser,0)
x_tenser_1=torch.unsqueeze(x_tenser,1)
print(
    '\n x',x,
    '\n x_tenser',x_tenser,
    '\n x_tenser.shape',x_tenser.shape,
    '\n x_tenser_0',x_tenser_0,
    '\n x_tenser_0.shape',x_tenser_0.shape,
    '\n x_tenser_1', x_tenser_1,
    '\n x_tenser_1.shape', x_tenser_1.shape
)