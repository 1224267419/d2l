import torch
import  numpy

x=torch.arange(12)
print("range",x)

x=torch.tensor([1,2,3,4])
#下面等价
y=torch.ones(4)*2
print(x**y)
print(x**2)

#创建时重置元素类型
x=torch.arange(12,dtype=torch.float32).reshape(3,4)
y=torch.tensor(
    [[2,1,4,3],
     [1,2,3,4],
     [4,3,2,1],
     [1,1,1,1]]
)
print(x.shape)
print(y.shape)

#第0维不变1,在其他维度合并
z=torch.cat((x,y),dim=0)
print(z.shape)
print(z)
#第一维度大小不同,因此不能合并
# z=torch.cat((x,y),dim=1)
# print(z.shape)
# print(z)

#广播机制:
a=torch.ones(2,1)
b=torch.ones(1,3)
c=a+b
print(c)
print(c.shape)

#直接操作数组的行或列
x=torch.zeros(2,3)
print("x=",x)
#所有行的0-1列修改为12
x[:,0:2]=12
print("x=",x)

#原地操作
x=torch.ones(1,2,3)
y=torch.ones(1,2,3)
before=id(y)
y[:]=y+x
# y+=x #和上面等价,也是原地操作

print(id(y)==before)
#非原地操作,和上面做对比
x=torch.ones(1,2,3)
y=torch.ones(1,2,3)
before=id(y)
#和上面不同,实际上是创建了新的y
y=y+x
print(id(y)==before)

#numpy转换为tensor
x=numpy.ones((1,2,3))
print(torch.tensor(x))

#tensor转换为numpy
x=torch.ones(1,2,3)
y=x.numpy()
print(y)

#复杂切片的情况
y=torch.ones(1,2,3)
y[:,0:1,:]=0
print(y)
y=torch.ones(1,2,3)
y[:,:,0:1]=0
print(y)

#按维度的求和操作
a=torch.arange(20).reshape(4,-1)
#保持维度数目不变的同时求和
sum_a=a.sum(axis=1,keepdim=True)#在第k个维度求和,从左往右
print("a.shape",a.shape)
print("sum_a.shape",sum_a.shape)
#通过广播机制利用sum
print(a/sum_a)
