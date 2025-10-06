import torch
import random
import d2l
print(torch.cuda.is_available())

def synthetic_data(w,b,num_examples):
    #生成y=wx+b
    x=torch.normal(0,1,(num_examples,len(w)))
    y=torch.matmul(x,w)+b
    y+=torch.normal(0,0.01,y.shape)
    #加入随机噪音
    return x,y.reshape((-1,1 ))
true_w=torch.tensor([2,-3.4])
true_b=4.2
feature,labels=synthetic_data(true_w,true_b,1000)