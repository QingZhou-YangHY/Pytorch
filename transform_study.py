from PIL import Image
from torch.utils.tensorboard import SummaryWriter

from torchvision import transforms

#python当中的用法  tensor数据类型  例:ToTensor
#通过transforms.ToTensor去解决两个问题
#1.transforms应该如何使用（python）
#2.为什么我们需要Tensor数据类型

#绝对路径 D:\Learn_Pytorch\dataset\train\people\OIP-C.jpg
#相对路径 dataset/train/people/OIP-C.jpg
img_path = "dataset/train/people/OIP-C.jpg"
img_path_abs = "D:\Learn_Pytorch\dataset\train\people\OIP-C.jpg"
#alt+enter可以直接导入你所需要的包
img = Image.open(img_path)

writer = SummaryWriter("logs")

#创建了ToTensor的对象
tensor_trans = transforms.ToTensor()

#ctrl+p 可以看需要什么参数
tensor_img = tensor_trans(img)
#现在将img的数据类型转化成了tensor
#解决了第一个问题，使用transforms的方法如上面例子所示
#详细步骤：从transforms当中选择一个class对他进行创建（transforms.ToTensor()），然后根据创建的工具传入参数就可以返回出结果

writer.add_image("Tensor_img",tensor_img)

writer.close()



print(tensor_img)