import nn
import torch
import torchvision

#引入所有的定义 *
from model_save import *

#方式1,保存方式1,加载模型
model = torch.load("vgg16_method1.pth")
#print(model)

#方式2,加载模型
vgg16 = torchvision.models.vgg16(pretrained=False)
vgg16.load_state_dict(torch.load("vgg16_method2.pth"))
#model = torch.load("vgg16_method2.pth")
#print(vgg16)

#陷阱1
#方式1:注意这里面要把自己定义的类也要复制过来,要不然会报错,他不知道往哪里面传参
#方式2:在上面把定义全都引入过来 from xxx import *

model = torch.load("hy_method1.pth")
print(model)