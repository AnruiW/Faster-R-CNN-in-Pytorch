import os
import torch
from PIL import Image
import pandas as pd
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from Object_Detection.Faster_RCNN_in_Pytorch.net.vgg import vgg_net
from Object_Detection.Faster_RCNN_in_Pytorch.net.rpn import rpn_net
from Object_Detection.Faster_RCNN_in_Pytorch.net.fasterrcnn import frcnn_net


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
        with open(r'C:\Users\Daybr\PycharmProjects\OneFlow\Object_Detection\Faster_RCNN_in_Pytorch\dataset\ImageNet_Label.txt', 'r', encoding='utf-8') as file:
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

    imagenet_train_set = datasets.ImageFolder(os.path.join(root_dir, 'mini_imagenet/train'), transform=image_transform)
    imagenet_val_set = datasets.ImageFolder(os.path.join(root_dir, 'mini_imagenet/val'), transform=image_transform)

    train_dataloader = DataLoader(imagenet_train_set, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(imagenet_val_set, batch_size=batch_size, shuffle=True)

    return train_dataloader, val_dataloader


def load_coco_dataset(root_dir, json_dir, batch_size=16):
    coco_dataset = datasets.CocoCaptions(root_dir, json_dir, transform=transforms.ToTensor())

    coco_train_set, coco_val_set = random_split(coco_dataset, [0.7, 0.3])

    train_dataloader = DataLoader(coco_train_set, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(coco_val_set, batch_size=batch_size, shuffle=True)

    return train_dataloader, val_dataloader


def load_VOC_dataset():
    pass


class VOC_Dataset(Dataset):
    def __init__(self, VOC_dir, year, is_train):
        self.size = (800, 800)
        VOC_transforms = transforms.Compose([
            transforms.Resize(self.size),
            transforms.ToTensor()
        ])
        self.VOC_dataset = datasets.VOCDetection(VOC_dir, year=year, image_set=is_train, download=True, transform=VOC_transforms)

        label_dict = {}
        with open(r"C:\Users\Daybr\PycharmProjects\OneFlow\Object_Detection\Faster_RCNN_in_Pytorch\dataset\VOC_2007_Class.txt") as f:
            for i in range(20):
                line = f.readline()
                label_dict[line[:-1]] = i
        self.VOC_label_dict = label_dict

    def __len__(self):
        return len(self.VOC_dataset)

    def __getitem__(self, item):
        image = self.VOC_dataset[item][0]

        annotation = self.VOC_dataset[item][1]['annotation']
        origin_size = annotation['size']
        object_list = annotation['object']

        x_scale = int(origin_size['width']) / self.size[0]
        y_scale = int(origin_size['height']) / self.size[1]

        gt_cat_list = []
        gt_bbox_list = []

        # convert single object into a list
        if isinstance(object_list, dict):
            object_list = [object_list]

        for obj in object_list:
            gt_cat_list.append(self.VOC_label_dict[obj['name']])
            x = (int(obj['bndbox']['xmin']) + int(obj['bndbox']['xmax'])) // 2
            y = (int(obj['bndbox']['ymin']) + int(obj['bndbox']['ymax'])) // 2
            w = int(obj['bndbox']['xmax']) - int(obj['bndbox']['xmin'])
            h = int(obj['bndbox']['ymax']) - int(obj['bndbox']['ymin'])
            # compute the coordinate after affine transformation
            gt_bbox_list.append(torch.tensor([x*x_scale, y*y_scale, w*x_scale, h*y_scale]))

        return image, gt_cat_list, gt_bbox_list


if __name__ == "__main__":
    voc_dataset = VOC_Dataset(r'D:\Dataset\VOC', '2007', 'train')
    voc_train = DataLoader(voc_dataset, batch_size=2, shuffle=True)

    vgg = vgg_net()
    frcnn = frcnn_net(vgg.get_feature_layer(), 20)

    for image, c, box in voc_train:
        print(c)
        print(box)
        print(image.shape)
        print(frcnn(image))

