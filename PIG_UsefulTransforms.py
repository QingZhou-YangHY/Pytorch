from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

writer = SummaryWriter("logs")
img = Image.open("D:\\Learn_Pytorch\\dataset\\train\\people\\OIP-C.jpg")
print(img)

#ToTensor
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
writer.add_image("ToTensor",img_tensor)

#Normalize(Normalize标准化)
print(img_tensor[0][0][0])
trans_norm = transforms.Normalize([1,3,5],[3,2,1])
img_norm = trans_norm(img_tensor)
print(img_norm[0][0][0])
writer.add_image("Normalize",img_norm,2)

#Resize
print(img.size)
trans_resize = transforms.Resize((512,512))
#img PIL ->resize -> img_resize PIL
img_resize = trans_resize(img)
#img_resize PIL -> totensor -> img_resize tensor
img_resize = trans_totensor(img_resize)
writer.add_image("Resize",img_resize,0)
print(img_resize)

#Compose - resize -2  Compose本质上就是把传进去的两个函数功能合并在一起了，但是前一个函数的返回值和下一个函数所需要的参数类型要一样/匹配
trans_resize_2 = transforms.Resize(512)
#PIL -> PIL -> tensor
trans_compose = transforms.Compose([trans_resize_2,trans_totensor])
img_resize_2 = trans_compose(img)
writer.add_image("Resize",img_resize_2,1)

#RandomCrop()   拿到一个新函数，要去看源码看档案知道怎么使用
trans_random = transforms.RandomCrop([100,100])
trans_compose_2 = transforms.Compose([trans_random,trans_totensor])
for i in range(10):
    img_crop = trans_compose_2(img)
    writer.add_image("RandomCrop",img_crop,i)


writer.close()