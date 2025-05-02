import torch
from nn import Sequential
from torch import nn
from torch.nn import Conv2d,MaxPool2d,Flatten,Linear
from torch.utils.tensorboard import SummaryWriter

##有很多东西大致上是一样的（构建神经网络，创建实例....包括后面写入，用tensorboard）
##Sequential和Compose本质一样，都是把好几个方法放在一块，让代码看起来更加简洁
##这里面都是nn.xxx是因为有tensorflow这个package，会被误认
#小技巧：1.有时候怕打错可以直接写nn.xxx让他自动补全直接tab
#      2.如果不知道卷积/池化的参数可以print一下看看是多少然后填进去

class HY(nn.Module):
    def __init__(self):
        super(HY,self).__init__()
        self.model1 = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(512, 64),
            nn.Linear(64, 10)
        )

    def forward(self,x):
        x = self.model1(x)
        return x

hy = HY()
print(hy)
input = torch.ones((64,3,32,32))
output = hy(input)
print(output.shape)

writer = SummaryWriter("logs_seq")
writer.add_graph(hy,input)
writer.close()