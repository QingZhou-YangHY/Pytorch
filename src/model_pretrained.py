import nn
import torchvision
from tensorflow.python.autograph.pyct.common_transformers.anf import transform

#train_data = torhcvision.datasets.ImageNet("../data_image_net", split = 'train', download = True,
#                                            transform=torchvision.transforms.ToTensor())
#因为这个数据集太大了(130G),并且要在网站上下载，用pip不行，所以换了一个模型，但本质一样

vgg16_false = torchvision.models.vgg16(pretrained=False)
vgg16_true = torchvision.models.vgg16(pretrained=True)

print(vgg16_true)

#../是上一层  ./是这一层
train_data = torchvision.transforms.ToTensor('../data',train=True,transform=torchvision.transforms.ToTensor(),
                                             download=True)
#在模型中加入自己想要的函数，想要加到哪层就.到哪里，比如这个.classifier
#下面呈现的就是加在两个不同的地方
vgg16_true.add_module('add_linear',nn.Linear(1000,10))
vgg16_true.classifier.add_module('add_linear',nn.linear(1000,10))
print(vgg16_true)

#修改模型中某一步变成自己想要的
vgg16_false.classifier[6] = nn.Linear(4096,10)
print(vgg16_false)