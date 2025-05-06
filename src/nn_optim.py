#优化器的简单使用
import torch
import torchvision
from torch import nn
from torch.nn import Sequential,Conv2d,MaxPool2d,Flatten,Linear
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("../data",train=False,transform=torchvision.transforms.ToTensor(),
                                       download=True)

dataloader = DataLoader(dataset,batch_size=1)

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

    def forward(self, x):
        x = self.model1(x)
        return x

loss = nn.CrossEntropyLoss()
hy = HY()
#其实优化器很简单，本质上就是套一层循环，然后这个循环里面不断地让loss变小
#这里面之所以两个循环，里面那个是过一遍整个数据集，外面的才是一遍一遍地优化，减小loss
#还是那句话！要想真正弄明白函数要多看官方文档
optim = torch.optim.SGD(hy.parameters(),lr=0.01)
for epoch in range(20):
    running_loss = 0.0
    for data in dataloader:
        imgs,targets = data
        outputs = hy(imgs)
        result_loss = loss(outputs,targets)
        optim.zero_grad()
        result_loss.backward()
        optim.step()
        running_loss = running_loss + result_loss
    print(running_loss)