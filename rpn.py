import torch
import torch.nn as nn
from Object_Detection.Faster_RCNN_in_Pytorch.vgg import vgg_net

import cv2
import numpy as np


def generate_region_proposal():
    pass


class rpn_net(nn.Module):
    def __init__(self, feature_layer, anchor_num=9):
        """
        :param anchor_num: the number of anchor generate at each location

        Every anchor will predict 2 scores estimating the probability of object or not object.
        Every anchor will predict 4 number corresponding to its location.
        """
        super(rpn_net, self).__init__()

        self.feature = feature_layer
        self.conv = nn.Conv2d(512, 512, 3, 1, 1)
        self.l_conv = nn.Conv2d(512, anchor_num * 2, 1, 1, 0)
        self.s_conv = nn.Conv2d(512, anchor_num * 4, 1, 1, 0)

    def forward(self, x):
        feature = self.feature(x)
        feature = self.conv(feature)

        anchor_location = self.l_conv(feature)
        score_list = self.s_conv(feature)
        return anchor_location, score_list


if __name__ == '__main__':
    image = torch.from_numpy(cv2.imread("test_image.jpg"))
    print(generate_anchor_box((7, 7)))

