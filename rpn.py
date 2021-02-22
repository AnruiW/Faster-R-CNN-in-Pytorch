import torch
import torch.nn as nn

import cv2


def generate_region_location():
    pass


class rpn(nn.Module):
    def __init__(self, anchor_num):
        """
        :param anchor_num: the number of anchor generate at each location

        Every anchor will predict 2 scores estimating the probability of object or not object.
        Every anchor will predict 4 number corresponding to its location.
        """
        super(rpn, self).__init__()

        self.conv = nn.Conv2d(512, 512, 3, 1, 1)
        self.l_conv = nn.Conv2d(512, anchor_num * 2, 1, 1, 0)
        self.s_conv = nn.Conv2d(512, anchor_num * 4, 1, 1, 0)

    def forward(self, input):
        input = self.conv(input)

        anchor_location = self.l_conv(input)
        location_list = 1

        score_list = self.s_conv(input)


if __name__ == '__main__':
    image = cv2.imread("test_image.jpg")
    print(image.shape)
    rpn_net = rpn(9)
    print(rpn_net(image))
