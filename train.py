import sys
sys.path.append(r"C:\Users\Daybr\PycharmProjects\OneFlow")

from Object_Detection.Faster_RCNN_in_Pytorch.vgg import vgg
from Object_Detection.Faster_RCNN_in_Pytorch.rpn import rpn

import torch
import numpy as np
import cv2


if __name__ == '__main__':
    image = cv2.imread("test_image.jpg")
    print(image.shape)
    vgg_net = vgg()
    output = vgg_net(torch.from_numpy(image).view([1, image.shape[0], image.shape[1], image.shape[2]]))
    print(output.shape)
    rpn_net = rpn(9)
    print(rpn_net(image))



