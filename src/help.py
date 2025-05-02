import torch

state_dict = torch.load('best_regression_model.pth', map_location=torch.device('cpu'))

print("Keys in state_dict:")
for key in state_dict.keys():
    print(key)

print("\nExample parameter values:")
for key, value in state_dict.items():
    print(f"{key}: {value.shape}")
    # 打印部分参数值（例如前5个元素）
    print(f"  Values: {value.flatten()[:5]}")
    break