import os
import torch
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import random

class BuildingDataset(Dataset):
    def __init__(self, directory):
        self.directory = directory
        self.images = []
        self.labels = []
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),  # 调整图片大小
            transforms.ToTensor(),  # 转换为Tensor
        ])

        # 遍历文件夹，加载图片
        for label_folder in os.listdir(directory):
            folder_path = os.path.join(directory, label_folder)
            if os.path.isdir(folder_path):
                for img_file in os.listdir(folder_path):
                    if img_file.endswith('.jpg') or img_file.endswith('.png'):
                        self.images.append(os.path.join(folder_path, img_file))
                        self.labels.append(label_folder)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')  # 确保图片为RGB格式

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        return image, label



# 创建数据集实例
dataset = BuildingDataset('D:\Record_BTC\Course\Pattern_Recoginition')

# # 创建数据加载器
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

def imshow(img):
    img = img.numpy().transpose((1, 2, 0))
    plt.imshow(img)
    plt.show()

# 随机选择几张图片进行展示
num_to_display = 4
indices = random.sample(range(len(dataset)), num_to_display)

for i in indices:
    image, label = dataset[i]
    print(f"Label: {label}")
    imshow(image)