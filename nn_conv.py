import torch
#输入的数据
import torch.nn.functional as F

input = torch.tensor([[1,2,0,3,1],
                     [0,1,2,3,1],
                     [1,2,1,0,0],
                     [5,2,3,1,1],
                     [2,1,0,1,1]])
#卷积核的数据
kernel = torch.tensor([[1,2,1],
                     [0,1,0],
                     [2,1,0]])

input = torch.reshape(input,(1,1,5,5))
kernel = torch.reshape(kernel,(1,1,3,3))

print(input.shape)
print(kernel.shape)

#stride：步径(一次走多少)，padding:一圈扩展多少(在输入的数据集里面扩展)
output = F.conv2d(input,kernel,stride = 1)
print(output)