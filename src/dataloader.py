#Dataloader的用法
import torchvision

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

#准备的测试数据集
test_data = torchvision.datasets.CIFAR10("./dataset",train=False,transform=torchvision.transforms.ToTensor())
#一般情况下发把shuffle设置为True(打乱)
test_loader = DataLoader(dataset=test_data,batch_size=64,shuffle=True,num_workers=0,drop_last=True)

#测试数据集中第一张图片及target
img,target = test_data[0]
print(img.shape)
print(target)

writer = SummaryWriter("../dataloader")

#对数据进行重新抓取
for epoch in range(2):
    step = 0
    #抓取了一轮数据
    #把imgs输送到神经网络，作为神经网络的一个输入
    for data in test_loader:
        imgs,targets = data
        # print(imgs.shape)
        # print(targets)
        writer.add_images("Epoch:{}".format(epoch),imgs,step)
        step = step + 1

writer.close()