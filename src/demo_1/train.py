import torch
import torchvision
from model import *
from torch.utils.tensorboard import SummaryWriter

#准备数据集
from torch import nn
from torch.utils.data import DataLoader


train_data = torchvision.datasets.CIFAR10(root="../../data",train=True,transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10(root="../../data",train=False,transform=torchvision.transforms.ToTensor(),
                                         download=True)

train_data_size = len(train_data)
test_data_size = len(test_data)

print("训练数据集的长度为{}".format(train_data_size))
print("测试数据集的长度为{}".format(test_data_size))

#利用 Dataloader 来加载数据
train_dataloader = DataLoader(train_data,batch_size=64)
test_dataloader = DataLoader(test_data,batch_size=64)

#创建网络
hy = HY()

#损失函数
loss_fn = nn.CrossEntropyLoss()

#优化器
#1e-2 = 1 x (10)^(-2) = 1 / 100 = 0.01
learning_rate= 1e-2
optimizer = torch.optim.SGD(hy.parameters(),lr=learning_rate)

#设置训练网络的一些参数
#记录训练次数
total_train_step = 0
#记录测试次数
total_test_step = 0
#训练的次数
epoch = 10

#添加tensorboard
writer = SummaryWriter("logs_train")

for i in range(epoch):
    print("----------第{}轮训练开始----------".format(i+1))

    #训练步骤开始
    hy.train()
    for data in train_dataloader:
        imgs,targets = data
        outputs = hy(imgs)
        loss = loss_fn(outputs,targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            print("训练次数:{},Loss = {}".format(total_train_step,loss.item()))
            writer.add_scalar("train_loss",loss.item(),total_train_step)

    #测试步骤开始
    hy.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs,targets = data
            outputs = hy(imgs)
            loss = loss_fn(outputs,targets)
            total_test_loss = total_test_loss + loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy

    print("整体测试集上的Loss:{}".format(total_test_loss))
    print("整体测试集上的正确率:{}".format(total_accuracy/test_data_size))
    writer.add_scalar("test_loss",total_test_loss,total_test_step)
    writer.add_scalar("test_accuracy",total_accuracy/test_data_size,total_test_step)
    total_test_step = total_test_step + 1

    torch.save(hy,"hy_{}.pth".format(i))
    print("模型已保存")

writer.close()