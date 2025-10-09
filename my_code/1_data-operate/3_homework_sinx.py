import torch
import matplotlib.pyplot as plt
x=torch.arange(1,10,0.01)
x.requires_grad_()

x1=x.detach()
y=torch.sin(x)
#为什么使用`.sum()`？因为`backward()`函数需要在一个标量上调用。如果y是一个向量（非标量），那么我们需要提供一个与y同形状的梯度权重向量（即`v`）作为`backward(v)`的参数。
# 但在这个例子中，我们并没有这样的权重，所以我们将y的所有元素求和，得到一个标量，然后对这个标量进行反向传播。这样，x的梯度就是y关于x的导数（即`dy/dx`）的累加
# 实际上，对于每个x_i，我们计算的是`d(sum(y))/dx_i = dy_i/dx_i`，因为y_i只与x_i有关，
# 而sum(y)对x_i的导数就是dy_i/dx_i。所以，`x.grad`将包含每个x_i处的导数`cos(x_i)`。
y.sum().backward()

plt.plot(x1,y.detach())
plt.plot(x1,x.grad)
plt.show()