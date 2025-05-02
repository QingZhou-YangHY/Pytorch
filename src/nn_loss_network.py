##损失函数（loss）与反向传播（backward），目的是用来优化，降低loss
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

##先输入nn.C（打个比方）后面就自动补全了，直接tab
##这里可以打断点然后debug一步一步看分析来了解具体过程
loss = nn.CrossEntropyLoss()
hy = HY()
for data in dataloader:
    imgs,targets = data
    outputs = hy(imgs)
    result_loss = loss(outputs,targets)
    print(result_loss)