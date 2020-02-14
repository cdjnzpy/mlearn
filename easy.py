import torch
from torch import nn
import numpy as np
torch.manual_seed(1)#设置随机数种子，使得结果确定

torch.set_default_tensor_type('torch.FloatTensor')#应该是设置默认数据结构
#set input number
num_inputs = 2
num_examples = 1000
true_w = [2,-3.4]
true_b = 4.2

features = torch.randn(num_examples,num_inputs,dtype=torch.float32)#生成1000*2变量每个分量
labels = true_w[0]*features[:,0]+true_w[1]*features[:,1]+true_b#按照最开始的几个方程构造我们的面案数据price
labels += torch.tensor(np.random.normal(0,0.01,size=labels.size()),dtype=torch.float32)#每一项随意加一个高斯分布函数中某个值

#读取数据集
batch_size =10


import torch.utils.data as Data
dataset = Data.TensorDataset(features,labels)
data_iter = Data.DataLoader(
    dataset = dataset,
    batch_size = batch_size,
    shuffle = True,#打乱
    num_workers = 2, #工作线程数
)

#定义模型
class LinearNet(nn.Module):
    def __init__(self,n_feature):
        super(LinearNet,self).__init__()
        self.linear = nn.Linear(n_feature,1)

    def forward(self,x):
        y = self.linear(x)
        return y
net = LinearNet(num_inputs)
print(net)


#三种方法来multilyer network
net = nn.Sequential(
    nn.Linear(num_inputs,1)
)

from torch.nn import init
init.normal_(net[0].weight,mean=0.0,std = 0.01)
init.constant_(net[0].bias,val = 0.0)
for param in net.parameters():
    pass
#损失


loss=nn.MSELoss
#定义优化函数
import torch.optim as optim
optimizer = optim.SGD(net.parameters(),lr=0.03)
print(optimizer)


num_epochs = 3
for epoch in range(1,num_epochs+1):
    for X,x in data_iter:
        output = net(X)
        l = loss(output,y.view(-1,1))
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
    print(epoch,l.item())
dense = net[0]
print(true_w,dense.weight.data)
print(true_b,dense.bias.data)




