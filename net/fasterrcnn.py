from torch import nn
import torch
import torchvision.ops.boxes as boxes
from Object_Detection.Faster_RCNN_in_Pytorch.utils.anchor import convert_rpn_output_predict_bbox


def extract_fature_map(feature_map, coordinate):
    return feature_map[:, int(coordinate[0]):int(coordinate[2]), int(coordinate[1]):int(coordinate[3])]


class frcnn_net(nn.Module):
    def __init__(self, rpn, num_class=20):
        super(frcnn_net, self).__init__()

        self.rpn = rpn
        self.softmax = nn.Softmax(dim=2)
        self.pool = nn.AdaptiveAvgPool2d((7, 7))
        self.fc = nn.Linear(7*7*512, 4096)
        self.l_fc = nn.Linear(4096, num_class * 4)
        self.s_fc = nn.Linear(4096, num_class + 1)

    def forward(self, x):
        feature_map, proposal_list, score_list = self.rpn(x)
        bbox_list = []
        for i in range(len(proposal_list)):
            bbox_list.append(convert_rpn_output_predict_bbox(proposal_list[i]))

        bbox_list = torch.stack(bbox_list)
        bbox_list[:, :, :2] = bbox_list[:, :, :2] - bbox_list[:, :, 2:] / 2
        bbox_list[:, :, 2:] = bbox_list[:, :, :2] + bbox_list[:, :, 2:]
        bbox_list = torch.clamp(bbox_list, 0, 800)

        score_list = self.softmax(score_list)
        tem_bbox_list = []
        for i in range(len(proposal_list)):
            bbox_index_list = boxes.nms(bbox_list[i], score_list[i, :, 1], 0.7)
            if self.training:
                tem_bbox_list.append(bbox_list[i][bbox_index_list, :][:500])
            else:
                tem_bbox_list.append(bbox_list[i][bbox_index_list, :][:1000])

        l_out_list = []
        s_out_list = []
        for i, batch_bbox in enumerate(tem_bbox_list):

            l_list = []
            s_list = []
            for bbox in batch_bbox:
                scale_bbox = bbox / 32
                mini_feature_map = extract_fature_map(feature_map[i], scale_bbox)

                pool = self.pool(mini_feature_map)
                fc = self.fc(torch.flatten(pool, 0))
                l_fc = self.l_fc(fc)
                s_fc = self.s_fc(fc)

                max_index = torch.argmax(s_fc)
                l_list.append(l_fc[max_index:max_index+4])
                s_list.append(s_fc)

            l_out_list.append(torch.stack(l_list))
            s_out_list.append(torch.stack(s_list))

        l_out_list = torch.stack(l_out_list)
        s_out_list = torch.stack(s_out_list)

        return l_out_list, s_out_list

