import torch
import numpy as np
import random
import matplotlib.pyplot as plt

#set input number
num_inputs = 2
num_examples = 1000
true_w = [2,-3.4]
true_b = 4.2

features = torch.randn(num_examples,num_inputs,dtype=torch.float32)#生成1000*2变量每个分量
labels = true_w[0]*features[:,0]+true_w[1]*features[:,1]+true_b#按照最开始的几个方程构造我们的面案数据price
labels += torch.tensor(np.random.normal(0,0.01,size=labels.size()),dtype=torch.float32)#每一项随意加一个高斯分布函数中某个值

#读入数据集
def data_iter(batch_size,features,labels):
    num_examples = len(features)
    indices  = list(range(num_examples))
    random.shuffle(indices)#随机排序
    for i in range(0,num_examples,batch_size):
        j = torch.LongTensor(indices[i:min(i+batch_size,num_examples)])
        yield features.index_select(0,j),labels.index_select(0,j) #按照行索引，每个索引j中的位置
'''

for X,y in data_iter(batch_size,features,labels):
    print(X,'\n',y)
    break
'''
batch_size = 10
#初始化模型参数
w = torch.tensor(np.random.normal(0,0.01,(num_inputs,1)),dtype=torch.float32)
b = torch.zeros(1,dtype=torch.float32)

w.requires_grad_(requires_grad = True)#节点 什么鬼叶子节点，如果对于什么求导的就需要T
b.requires_grad_(requires_grad = True)

#定义模型
def linreg(X,w,b):
    return torch.mm(X,w)+b#矩阵相乘

#损失函数
def squared_loss(y_hat, y):
    return (y_hat-y.view(y_hat.shape))**2/2#view修改tensor的形状

#定义优化函数(小批量随机梯度下降)
def sgd(params,lr,batch_size):
    for param in params:
        param.data -= lr*param.grad/batch_size

#设置参数准备训练模型啦
lr = 0.03
num_epochs = 5  #训练次数
#换个名字
net = linreg        #方程
loss = squared_loss #loss函数
for epoch in range(num_epochs):
    for X,y in data_iter(batch_size,features,labels):
        l = loss(net(X,w,b),y).sum()
        l.backward()#貌似是梯度求解，每一次计算智能bw一次
        #原解释:calculate the gradient of batch sample loss
        sgd([w,b],lr,batch_size)
        w.grad.data.zero_()
        b.grad.data.zero_()#情空
    train_l = loss(net(features,w,b),labels)
    print("number{}times,lossfor{}".format(epoch+1,train_l.mean().item()))

print(w,true_w,"\n",b,true_b)