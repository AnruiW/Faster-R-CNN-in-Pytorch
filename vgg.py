import torch
import torch.nn as nn
from torchvision.models import vgg16


def compose_layers(net_config, in_channels, use_bn):
    layers = []
    for layer in net_config:
        if layer == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels=in_channels, out_channels=layer, kernel_size=3, stride=1, padding=1)
            if use_bn:
                layers += [conv2d, nn.BatchNorm2d(layer), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = layer
    return nn.Sequential(*layers)


# the Conv part of VGG net
def vgg_feature(net_type='D', in_channels=3, use_bn=False):
    ConvNet_Configuration = {
        'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
        'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    }

    return compose_layers(ConvNet_Configuration[net_type], in_channels, use_bn)


# the complete version of VGG net
class vgg_net(nn.Module):
    def __init__(self):
        super(vgg_net, self).__init__()

        self.feature = vgg_feature()
        self.fc = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 1000)
        )

    def forward(self, x):
        feature = self.feature(x)
        fc = self.fc(torch.flatten(feature, 1))
        return fc

    def get_feature_layer(self):
        return self.feature


if __name__ == '__main__':
    pass

