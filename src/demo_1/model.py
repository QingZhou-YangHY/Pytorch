import torch
from torch import nn

class HY(nn.Module):
    def __init__(self):
        super(HY,self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3,32,5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,32,5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,5,1,2),
            nn.Flatten(),
            nn.Linear(64*8*8,64),
            nn.Linear(64,10)
        )

    def forward(self,x):
        x = self.model(x)
        return x

if __name__ == '__main__':
    hy = HY()
    input = torch.ones((64,3,32,32))
    output = hy(input)
    print(output.shape)