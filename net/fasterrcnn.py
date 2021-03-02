from torch import nn
import torch

class frcnn_net(nn.Module):
    def __init__(self, feature, num_class):
        super(frcnn_net, self).__init__()
        self.feature = feature
        self.pool = nn.AdaptiveAvgPool2d((7, 7))
        self.fc = nn.Linear(7*7*512, 4096)
        self.l_fc = nn.Linear(4096, num_class * 4)
        self.s_fc = nn.Linear(4096, num_class)

    def forward(self, x):
        feature = self.feature(x)
        pool = self.pool(feature)
        fc = self.fc(torch.flatten(pool, 1))
        l_fc = self.l_fc(fc)
        s_fc = self.s_fc(fc)

        return l_fc, s_fc

    def get_feature_layer(self):
        return self.feature


