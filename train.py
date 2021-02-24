import sys
import os
sys.path.append(r"C:\Users\Daybr\PycharmProjects\OneFlow")
from torch import optim
from torch.utils.data import DataLoader, Dataset
from Object_Detection.Faster_RCNN_in_Pytorch.vgg import vgg_net
from Object_Detection.Faster_RCNN_in_Pytorch.rpn import rpn_net
import Object_Detection.Faster_RCNN_in_Pytorch.dataset as load_dataset
import torch
import numpy as np
import cv2
import time


def evaluate_val_accuracy(net, test_iter):
    net.eval()
    num_image, accuracy = 0, 0

    for test_image, test_label in test_iter:
        with torch.no_grad:
            accuracy += (net(test_image.to(device)).argmax(dim=1) == test_label.to(device)).float().sum().cpu().item()
        num_image += len(test_image)

    net.train()
    return accuracy / num_image


def pretrain_vgg(vgg, train_iter, test_iter, optimizer, lr_scheduler, num_epoch, device):
    vgg = vgg.to(device)
    print("pre-train VGG net on {}".format(device))
    start_time = time.time()
    checkpoint_dir = os.path.join(os.path.abspath('.'), 'vgg_checkpoint.pth')

    criteria = torch.nn.CrossEntropyLoss()

    if os.path.exists(checkpoint_dir):
        vgg.load_state_dict(torch.load(checkpoint_dir))
        print("successfully load model parameters")

    for epoch in range(num_epoch):
        epoch_loss = 0
        for image_list, label_list in train_iter:
            image_list.to(device)
            label_list.to(device)
            output_list = vgg(image_list)
            loss = criteria(output_list, label_list)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step(loss)
            epoch_loss += loss.cpu().item()
        val_accuracy = evaluate_val_accuracy(vgg, test_iter)
        print(f"epoch: {epoch+1}, training loss: {epoch_loss}, validation accuracy: {val_accuracy}, total training time: {time.time()-start_time}")

        torch.save(vgg, checkpoint_dir)
        print(f"saving model to {checkpoint_dir}")

    torch.save(vgg, os.path.join(os.path.abspath(), 'vgg_checkpoint.pth'))
    print(f"saving final model to {os.path.join(os.path.abspath(), 'vgg_checkpoint.pth')}")


if __name__ == '__main__':
    device = "gpu" if torch.cuda.is_available() else "cpu"

    vgg = vgg_net()
    vgg_lr = 1e-4
    vgg_optimizer = optim.Adam(vgg.parameters(), vgg_lr, weight_decay=5e-4)
    vgg_lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(vgg_optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    # vgg_train_set, vgg_val_set = load_dataset.load_imagenet_dataset('/ImageNet')
    mini_imagenet_train = load_dataset.mini_imagenet(r'D:\BaiduNetdiskDownload\mini-imagenet\mini-imagenet\images', 'train.csv')
    mini_imagenet_val = load_dataset.mini_imagenet(r'D:\BaiduNetdiskDownload\mini-imagenet\mini-imagenet\images', 'val.csv')

    vgg_train_set = DataLoader(mini_imagenet_train, batch_size=4, shuffle=True)
    vgg_val_set = DataLoader(mini_imagenet_val, batch_size=4, shuffle=True)

    pretrain_vgg(vgg, vgg_train_set, vgg_val_set, vgg_optimizer, vgg_lr_scheduler, 75, device)

    rpn = rpn_net(vgg.get_feature_layer())




