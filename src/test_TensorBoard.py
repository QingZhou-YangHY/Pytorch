from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

writer = SummaryWriter("../logs")

image_path = "../dataset/train/people/ynZMg18RfjSAEm7qAAtjHUPtSl0522.jpg"
img_PIL = Image.open(image_path)
img_array = np.array(img_PIL)
print(type(img_array))
print(img_array.shape)


writer.add_image("test",img_array,2,dataformats='HWC')
#y = x
for i in range(100):
    writer.add_scalar("y=2x",2*i,i)

writer.close()

