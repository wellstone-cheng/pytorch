import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
x=torch.linspace(-5,5,200) #
x_np=x.numpy()
#print(
#    '\n x',x,
#    '\n x_np',x_np
#)

y_relu=F.relu(x).numpy()
y_sigmoid =torch.sigmoid(x).numpy()     #        F.sigmoid(x).numpy()
y_tanh =torch.tanh(x).numpy()           #F.tanh(x).numpy()
y_softplus =F.softplus(x).numpy()
# y_softmax =F.softmax(x,dim=0).numpy

plt.figure(1,figsize=(8,6))             # 图片编号为1,尺寸为宽高8*6英寸,编号相同即在同一张图片上显示
plt.subplot(221)
plt.plot(x_np,y_relu,c='red',label='relu')
plt.ylim(-1,5)                          #y轴的范围为-1到5
plt.legend(loc='best')                  #给图加上图例,loc为位置

plt.subplot(222)
plt.plot(x_np,y_sigmoid,c='red',label='sigmoid')
plt.ylim(-0.1,1.1)                      #y轴的范围为-0.1到1.1
plt.legend(loc='best')                  #给图加上图例,loc为位置

plt.subplot(223)
plt.plot(x_np,y_tanh,c='red',label='tanh')
plt.ylim(-1.1,1.1)                        #y轴的范围为-1.1到1.1
plt.legend(loc='best')                    #给图加上图例,loc为位置

plt.subplot(224)
plt.plot(x_np,y_softplus,c='red',label='softplus')
plt.ylim(-0.1,6)                          #y轴的范围为-1.1到1.1
plt.legend(loc='upper left')              #给图加上图例,loc为位置

# plt.subplot(224)
# plt.plot(x_np,y_softmax,c='red',label='softmax')
# #plt.ylim(-0.1,6)                          #y轴的范围为-1.1到1.1
# plt.legend(loc='upper left')              #给图加上图例,loc为位置

plt.show()