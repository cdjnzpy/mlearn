import matplotlib.pyplot as plt
from IPython import display
import torch
import torchvision
import torchvision.transforms as transforms
import time
'''
import sys
sys.path.append("/home/kesci/input")
'''#这里是调用模块的路径
#设置一个路径
import os
path = os.path.dirname(__file__)

#import d2lzh1981 as d2l

#mnist_train = torchvision.datasets.FashionMNIST(root=path+'\\', train=True, download=True, transform=transforms.ToTensor())
#mnist_test = torchvision.datasets.FashionMNIST(root=path+'\\', train=False, download=True, transform=transforms.ToTensor())


###这一部分不是很清楚，缺少了必要的库

##多层感知机部分
##将上列的两层的网络家省委多层，包含一层隐藏单元，在输入与输出之间存在一个根伟复杂的计算
##也可以认为是将输入端层构造更多的输入并用此来表现出我们的不同输出层

def xyplot(x_vals,y_vals,name):#构造一个打出xy关系图的函数
    plt.plot(x_vals.detach().numpy(),y_vals.detach().numpy())#输入的数字类型进行改变
    plt.xlabel("x")
    plt.ylabel(name+"x")

x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)#设定x的分布范围


##一系列激活函数，进行变换的函数
#线性与非线性变换
#ReLU为通用激活，一个简单非线性变换，可类似于abs

y = x.relu()    #整活应该是一个abs函数
xyplot(x, y, 'relu')
plt.show()

#计算函数的导数，则y部分会变化
y.sum().backward()#计算偏导（这里就是导数了）
xyplot(x, x.grad, 'grad of relu')
plt.show()

#sigmoid函数，将数值变化在0-1之间，有点像前面分类算法中的那个三个调参的函数
y = x.sigmoid() #将我们已经已经设定范围的x压缩到0-1之间
xyplot(x, y, 'sigmoid')
plt.show()

#再试试对项目的函数求梯度
x.grad.zero_()#改变属性
y.sum().backward()
xyplot(x, x.grad, 'grad of sigmoid')
plt.show()

#在更换为其他函数来整活，然后求导（主要是熟悉一下函数）
y = x.tanh()
xyplot(x, y, 'tanh')
plt.show()

x.grad.zero_()
y.sum().backward()
xyplot(x, x.grad, 'grad of tanh')
plt.show()



#多层感知机的从零开始
#缺少数据参数集



