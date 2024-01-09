import os
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile
from sklearn.model_selection import train_test_split

ImageFile.LOAD_TRUNCATED_IMAGES = True  # 允许加载损坏的图像

class BuildingDataset(Dataset):
    """Custom Dataset for loading building images and their labels"""

    def __init__(self, image_dir, label_dir, transform=None, image_filenames=None, label_filenames=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.image_filenames = image_filenames or os.listdir(image_dir)
        self.label_filenames = label_filenames or os.listdir(label_dir)

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        while True:
            try:
                img_name = os.path.join(self.image_dir, self.image_filenames[idx])
                label_name = os.path.join(self.label_dir, self.label_filenames[idx])
                label_name = label_name.split('/')[-1]

                image = Image.open(img_name).convert('RGB')
                if self.transform:
                    image = self.transform(image)
                
                label = label_name
                return image, label

            except (IOError, OSError):
                print(f"Error loading image {img_name}. Skipping.")
                idx += 1  # Skip to the next index

                # Check if the new index is out of bounds
                if idx >= len(self.image_filenames):
                    return None, None
    # def __getitem__(self, idx):
    #     img_name = os.path.join(self.image_dir, self.image_filenames[idx])
    #     label_name = os.path.join(self.label_dir, self.label_filenames[idx])
    #     label_name = label_name.split('/')[-1]

    #     try:
    #         image = Image.open(img_name).convert('RGB')
    #         if self.transform:
    #             image = self.transform(image)
    #     except (IOError, OSError):
    #         print(f"Error loading image {img_name}. Skipping.")
    #         return None, None
    
    #     label = label_name

    #     return image, label

def get_data_loaders(image_dir, label_dir, batch_size, transform, test_size=0.2, shuffle=True, num_workers=0):
    image_filenames = []
    label_filenames = []
    # 遍历文件夹，加载图片
    for label_folder in os.listdir(image_dir):
        folder_path = os.path.join(image_dir, label_folder)
        if os.path.isdir(folder_path):
            for img_file in os.listdir(folder_path):
                if img_file.endswith('.jpg') or img_file.endswith('.png'):
                    image_filenames.append(os.path.join(folder_path, img_file))
                    label_filenames.append(label_folder)


    # Split into train and test sets
    image_filenames_train, image_filenames_test, label_filenames_train, label_filenames_test = train_test_split(
        image_filenames, label_filenames, test_size=test_size, random_state=42)

    train_dataset = BuildingDataset(image_dir, label_dir, transform, image_filenames_train, label_filenames_train)
    test_dataset = BuildingDataset(image_dir, label_dir, transform, image_filenames_test, label_filenames_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader
