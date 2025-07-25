[1_torch_tensor.py](../code/1_data-operate/1_torch_tensor.py)
reshape用于修改tensor的形状
广播机制:[1,3]+[2,1]=[2,3],仅对大小为1的维度有效
通过 `x[:,0:2]=12`的类数组赋值操作对于tensor也是可行的
对于特别大的数组,建议采用原地操作,如`y[:]=y+x`或`y+=x`,而`y=y+x`并非原地操作,而是创建了新的y

```python
#复杂切片的情况
y=torch.ones(1,2,3)
y[:,0:1,:]=0
print(y)
y=torch.ones(1,2,3)
y[:,:,0:1]=0
print(y)
```

根据上述例子理解一下切片位置和高维矩阵的定义



#### 矩阵微积分

![1752673366563](./1_data-operate.assets\1752673366563.png)

**标量**对**列向量求导**的结果为**Y**对**X**各元素求导组成的行向量

![image-20250716214823928](./1_data-operate.assets\image-20250716214823928.png)

相反**列向量对标量求导结果还是列向量**

![image-20250716214855551](./1_data-operate.assets\image-20250716214855551.png)

向量**y**对向量**x**求导:如上图所示,将y拆解为多个标量单独求导即可

下面是一些常见的矩阵**微分表**

![image-20250716215127350](./1_data-operate.assets\image-20250716215127350.png)

##### 计算图

计算图即数据流图,在计算图中，每个节点都代表一个变量或操作，通过将计算过程表示为计算图，可以方便地计算各个变量对损失函数的导数，从而实现反向传播算法，使得深度神经网络的训练变得更加高效和简单。 
针对分布式计算的优化：计算图中的节点和边可以方便地划分到不同的计算节点或设备上进行分布式计算，从而优化计算性能。 
支持动态图和静态图：AI框架中的计算图既可以是静态图，也可以是动态图。
静态图在模型训练之前需要先定义好计算图，然后才能进行模型训练；而动态图则可以根据实际输入的数据动态地构建计算图，从而适应更加复杂的模型。 

显示计算：先给公式再给值；
隐式计算：先给值再给公式

##### 自动求导:

![image-20250716222458597](./1_data-operate.assets\image-20250716222458597.png)

反向传播更为常见,因为正向积累需要保存



#### 自动求导

代码给出了

 [2_auto_grad.py](..\code\1_data-operate\2_auto_grad.py) 

**多个loss**分别进行反向求导时需要累加梯度,**因此torch默认进行梯度累积**



[todo](https://www.bilibili.com/video/BV1PX4y1g7KC?spm_id_from=333.788.recommend_more_video.0&vd_source=82d188e70a66018d5a366d01b4858dc1)