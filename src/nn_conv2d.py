import torch
import torchvision
from torch import nn
from torch.nn import Conv2d
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("../data",train=False,transform=torchvision.transforms.ToTensor(),
                                       download = True)


dataloader = DataLoader(dataset,batch_size = 64)

class HY(nn.Module):
    def __init__(self):
        super(HY,self).__init__()
        self.conv1 = Conv2d(in_channels=3,out_channels=6,kernel_size=3,stride=1,padding=0)
    def forward(self,x):
        x = self.conv1(x)
        return x

hy = HY()

writer = SummaryWriter("../../logs")

step = 0

for data in dataloader:
    imgs,targets = data
    output = hy(imgs)
    print(imgs.shape)
    print(output.shape)
    #torch.Size([64,3,32,32])
    writer.add_images("input",imgs,step)
    #torch.Size([64,3,32,32]]) -> [xxx,3,30,30]

    output = torch.reshape(output,(-1,3,30,30))
    writer.add_images("output",output,step)


    step = step + 1


