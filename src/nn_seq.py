import torch
from nn import Sequential
from torch import nn
from torch.nn import Conv2d,MaxPool2d,Flatten,Linear
from torch.utils.tensorboard import SummaryWriter


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