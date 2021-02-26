from torch import optim
from torch.utils.data import DataLoader
from Object_Detection.Faster_RCNN_in_Pytorch.net.vgg import vgg_net
from Object_Detection.Faster_RCNN_in_Pytorch.net.rpn import rpn_net
from Object_Detection.Faster_RCNN_in_Pytorch.net.fasterrcnn import frcnn_net
import Object_Detection.Faster_RCNN_in_Pytorch.utils.dataset as dataset
from Object_Detection.Faster_RCNN_in_Pytorch.utils.sub_train import train_vgg, train_fasterrcnn
import torch
import time


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    vgg = vgg_net()
    vgg_lr = 1e-4
    vgg_optimizer = optim.SGD(vgg.parameters(), vgg_lr, momentum=0.9, weight_decay=5e-4)
    vgg_lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(vgg_optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    # vgg_train_set, vgg_val_set = load_dataset.load_imagenet_dataset('/ImageNet')
    mini_imagenet_train = dataset.mini_imagenet(r'D:\BaiduNetdiskDownload\mini-imagenet\mini-imagenet\images', '/dataset/train_3k.csv')
    mini_imagenet_val = dataset.mini_imagenet(r'D:\BaiduNetdiskDownload\mini-imagenet\mini-imagenet\images', '/dataset/val_1k.csv')

    vgg_train_set = DataLoader(mini_imagenet_train, batch_size=4, shuffle=True)
    vgg_val_set = DataLoader(mini_imagenet_val, batch_size=4, shuffle=True)

    train_vgg(vgg, vgg_train_set, vgg_val_set, vgg_optimizer, vgg_lr_scheduler, 75, device)

    rpn = rpn_net()
    frcnn = frcnn_net(vgg.get_feature_layer())
    rpn_lr = 1e-3
    frcnn_lr = 1e-4

    coco_train =
    coco_val =

    frcnn_train_set =
    frcnn_val_set =

    train_fasterrcnn(frcnn, rpn, rpn_lr, frcnn_lr, frcnn, device)



    coco_






if __name__ == "__main__":
    pass


