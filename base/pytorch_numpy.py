import torch
import numpy as np
np_date_1=np.arange(4).reshape(4,)#1维数组
np_date_2=np.arange(6).reshape(2,3)# numpy格式 取6个数据,分为2组,每组3个数据--2维数组
np_date_3=np.arange(24).reshape(4,3,2)# 3维数组
torch_data_1=torch.from_numpy(np_date_1)# 将numpy数据转成tensor格式
tensor2array_1= torch_data_1.numpy()#将Tensor格式转成numpy格式
torch_data_2=torch.from_numpy(np_date_2)# 将numpy数据转成tensor格式
tensor2array_2= torch_data_2.numpy()#将Tensor格式转成numpy格式
torch_data_3=torch.from_numpy(np_date_3)# 将numpy数据转成tensor格式
tensor2array_3= torch_data_3.numpy()#将Tensor格式转成numpy格式
print('\n numpy np_date_1',np_date_1)
print('\n torch_data_1',torch_data_1)
print('\n tensor2array_1',tensor2array_1)
print('\n numpy np_date_2',np_date_2)
print('\n torch_data_2',torch_data_2)
print('\n tensor2array_2',tensor2array_2)
print('\n numpy np_date_3',np_date_3)
print('\n torch_data_3',torch_data_3)
print('\n tensor2array_3',tensor2array_3)