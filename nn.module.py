#搭建一个神经网络
import torch
from torch import nn

#调用的时候才会运行
class YangHY(nn.Module):
    #一种定义方式
    def __init__(self):
        super(YangHY,self).__init__()
    def forward(self,input):
        output = input + 1
        return output

#创建了一个HY神经网络
HY = YangHY()

x = torch.tensor(1.0)

output = HY(x)

print(output)