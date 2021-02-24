import os
import torch
from PIL import Image
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import torchvision.datasets as dataset
import torchvision.transforms as transforms


class mini_imagenet(Dataset):
    def __init__(self, root_dir, csv_dir):
        self.root_dir = root_dir
        self.dir = csv_dir
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        imagenet_label_dict = {}
        with open('ImageNet_Label.txt', 'r', encoding='utf-8') as file:
            for i in range(1000):
                line = file.readline()[:-1].split(',')
                label_code, label_name = line[0], line[1:]
                imagenet_label_dict[label_code] = i
        self.image_label_dict = imagenet_label_dict

    def __len__(self):
        csv_file = pd.read_csv(self.dir)
        return csv_file.shape[0]

    def __getitem__(self, item):
        csv_file = pd.read_csv(self.dir)
        image_name = csv_file.iloc[item, 0]
        image_label = csv_file.iloc[item, 1]
        image = Image.open(os.path.join(self.root_dir, image_name))

        return self.transform(image), self.image_label_dict[image_label]


def load_imagenet_dataset(root_dir, batch_size=16):
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    imagenet_train_set = dataset.ImageFolder(os.path.join(root_dir, 'mini_imagenet/train'), transform=image_transform)
    imagenet_val_set = dataset.ImageFolder(os.path.join(root_dir, 'mini_imagenet/val'), transform=image_transform)

    train_dataloader = DataLoader(imagenet_train_set, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(imagenet_val_set, batch_size=batch_size, shuffle=True)

    return train_dataloader, val_dataloader


def load_coco_dataset():
    pass


if __name__ == "__main__":
    csv = pd.read_csv("val.csv")
    print(csv)
    print(csv.shape)
