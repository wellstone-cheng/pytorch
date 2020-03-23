import numpy as np
import torch
data=[[1, 2],[3, 4]]# 列表list
tensor=torch.Tensor(data) #tensor

print(
    '\n np.matmul', np.matmul(data,data),#矩阵乘法
    '\n np.multiply', np.multiply(data,data),#矩阵点乘
    '\n np.dot', np.dot(data,data),#矩阵叉乘 =矩阵乘法
    '\n torch.mm',torch.mm(tensor,tensor), #矩阵乘法
    '\n torch.mul',torch.mul(tensor,tensor), #矩阵点乘
    '\n torch.dot',torch.dot(torch.from_numpy(np.array(data).flatten()),torch.from_numpy(np.array(data).flatten())) # torch.dot 只能适用于1维的矩阵
)