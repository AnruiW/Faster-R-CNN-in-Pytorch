import torch
import torch.nn as nn
from torch.nn import functional as F
from Object_Detection.Faster_RCNN_in_Pytorch.net.vgg import vgg_net

import cv2
import numpy as np


def generate_region_proposal(location_list, score_list):
    print(location_list.shape)
    print(score_list.shape)



class rpn_net(nn.Module):
    def __init__(self, feature_layer, anchor_num=9):
        """
        :param anchor_num: the number of anchor generate at each location

        Every anchor will predict 2 scores estimating the probability of object or not object.
        Every anchor will predict 4 number corresponding to its location.
        """
        super(rpn_net, self).__init__()

        self.feature = feature_layer
        self.conv = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU()
        )
        self.l_conv = nn.Conv2d(512, anchor_num * 4, 1, 1, 0)
        self.s_conv = nn.Conv2d(512, anchor_num * 2, 1, 1, 0)

    def forward(self, x):
        feature = self.feature(x)
        rpn_feature = self.conv(feature)

        location_list = self.l_conv(rpn_feature)
        # the output of s_conv is (batch_size, 36, 7, 7)
        # need to be converted into ()
        location_list = location_list.permute(0, 2, 3, 1).contiguous().view(x.shape[0], -1, 4)


        score_list = self.s_conv(rpn_feature)
        # the output of s_conv is (batch_size, 18, 7, 7)
        # need to be converted into (batch_size, num of box, 2)
        score_list = score_list.permute(0, 2, 3, 1).contiguous().view(x.shape[0], -1, 2)

        return feature, location_list, score_list


if __name__ == '__main__':
    image = cv2.imread("test_image.jpg")
    image = cv2.resize(image, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
    print(image.shape)
    image = image.reshape((3, 224, 224))
    image = torch.from_numpy(image[np.newaxis, :], )
    image = image.float()
    print(image.shape)
    vgg = vgg_net()
    rpn = rpn_net(vgg.get_feature_layer())
    location_list, score_list = rpn(image)
    generate_region_proposal(location_list, score_list)

