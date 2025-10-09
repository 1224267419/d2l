import torch

x = torch.arange(4.0)
print(x)
#x实际上是一个实例,为其添加上求导
x.requires_grad_(True)
print(x)
#x自己做内积
y=2*torch.dot(x,x)
print(y)

#调用反向传播函数自动计算y关于x每个分量的梯度
y.backward()
print(x.grad)
#这里每一个y=2*sum(x_i^2),因此x.gard=2*2*x_i=4x_i

#torch默认会累积梯度,因此再次计算时需要梯度清零
#尝试注销下面这行,对比两次梯度的不同
x.grad.zero_()
y=x.sum()
y.backward()
print("梯度清零",x.grad)


#标量乘法求导
x.grad.zero_()
y=x*x
u=y.detach()#detach将变量转换为标量
z=u*x
z.sum().backward()
print("乘法求导",x.grad==u)