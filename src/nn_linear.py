import torch
import torchvision
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("../data",train=False,transform=torchvision.transforms.ToTensor(),
                                    download=True)

dataloader = DataLoader(dataset,batch_size=64)

class DY(nn.Module):
    def __init__(self):
        super(DY,self).__init__()
        self.linear1 = Linear(196608,10)

    def forward(self,input):
        self.linear1(input)
        return output
dy = DY()

for data in dataloader:
    imgs,targets = data
    print(imgs.shape)
    output = torch.flatten(imgs)
    print(output.shape)
    output = dy(output)
    print(output.shape)
