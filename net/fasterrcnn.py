from torch import nn


class faster_r_cnn(nn.Module):
    def __init__(self, feature, num_class):
        super(faster_r_cnn, self).__init__()
        self.feature = feature
        self.pool = nn.AdaptiveAvgPool2d((7, 7))
        self.fc = nn.Linear(7*7*512, 4096)
        self.l_fc = nn.Linear(4096, num_class * 4)
        self.s_fc = nn.Linear(4096, num_class)

    def forward(self, x):
        feature = self.feature(x)
        pool = self.pool(feature)
        fc = self.fc(pool)
        l_fc = self.l_fc(fc)
        s_fc = self.s_fc(fc)

        return l_fc, s_fc


