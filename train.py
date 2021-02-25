from torch import optim
from torch.utils.data import DataLoader
from Object_Detection.Faster_RCNN_in_Pytorch.net.vgg import vgg_net
from Object_Detection.Faster_RCNN_in_Pytorch.net.rpn import rpn_net
import Object_Detection.Faster_RCNN_in_Pytorch.utils.dataset as load_dataset
from Object_Detection.Faster_RCNN_in_Pytorch.utils.sub_train import train_vgg, train_rpn
import torch
import time


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"

    vgg = vgg_net()
    vgg_lr = 1e-4
    vgg_optimizer = optim.Adam(vgg.parameters(), vgg_lr, weight_decay=5e-4)
    vgg_lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(vgg_optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    # vgg_train_set, vgg_val_set = load_dataset.load_imagenet_dataset('/ImageNet')
    mini_imagenet_train = load_dataset.mini_imagenet(r'D:\BaiduNetdiskDownload\mini-imagenet\mini-imagenet\images', '/dataset/train_3k.csv')
    mini_imagenet_val = load_dataset.mini_imagenet(r'D:\BaiduNetdiskDownload\mini-imagenet\mini-imagenet\images', '/dataset/val_1k.csv')

    vgg_train_set = DataLoader(mini_imagenet_train, batch_size=4, shuffle=True)
    vgg_val_set = DataLoader(mini_imagenet_val, batch_size=4, shuffle=True)

    train_vgg(vgg, vgg_train_set, vgg_val_set, vgg_optimizer, vgg_lr_scheduler, 75, device)

    rpn = rpn_net(vgg.get_feature_layer())




