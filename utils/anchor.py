import torch
import cv2
import numpy as np
from torch.nn import LSTM


def compute_rpn_gt_output(anchor_bbox, gt_bbox):
    '''
    :param anchor_bbox: in the form of x,y,w,h
    :param gt_bbox: coordinate and WH for a ground-truth box, in the form of x,y,w,h
    :return: compute the ground-truth for the output of the bbox regression
    '''
    tx = (gt_bbox[0] - anchor_bbox[0]) / anchor_bbox[2]
    ty = (gt_bbox[1] - anchor_bbox[1]) / anchor_bbox[3]
    tw = np.log(gt_bbox[2] / anchor_bbox[2])
    th = np.log(gt_bbox[3] / anchor_bbox[3])

    result = torch.tensor([tx, ty, tw, th], dtype=torch.float32, requires_grad=False)
    return result


def convert_rpn_output_predict_bbox(rpn_output):
    '''
    :param rpn_output: the output of rpn location output
    :param anchor_bbox: corresponding anchor box
    :return: convert rpn location output to bbox coordinate
    '''
    anchor_bbox = generate_anchor_box((25, 25))
    anchor_bbox = anchor_bbox.float()

    x = rpn_output[:, 0] * anchor_bbox[:, 2] + anchor_bbox[:, 0]
    y = rpn_output[:, 1] * anchor_bbox[:, 3] + anchor_bbox[:, 1]
    w = torch.exp(rpn_output[:, 2]) * anchor_bbox[:, 2]
    h = torch.exp(rpn_output[:, 3]) * anchor_bbox[:, 3]

    rpn_anchor_bbox = torch.stack([x, y, w, h]).transpose(1, 0)
    # if is_train:
    #     low_bound = rpn_anchor_bbox[:, :2] - rpn_anchor_bbox[:, 2:] / 2
    #     del_index = np.where(low_bound < 0)[0]
    #     rpn_anchor_bbox = np.delete(rpn_anchor_bbox, del_index, axis=0)
    #
    #     high_bound = rpn_anchor_bbox[:, :2] + rpn_anchor_bbox[:, 2:] / 2
    #     del_index = np.where(high_bound > 800)[0]
    #     rpn_anchor_bbox = np.delete(rpn_anchor_bbox, del_index, axis=0)

    return rpn_anchor_bbox


def compute_gt_output(anchor_bbox, gt_bbox):
    '''
    :param anchor_bbox: in the form of x,y,w,h
    :param gt_bbox: coordinate and WH for a ground-truth box, in the form of x,y,w,h
    :return: the ground-truth for the output of RPN network
    '''
    tx = (gt_bbox[:, 0] - anchor_bbox[:, 0]) / anchor_bbox[:, 2]
    ty = (gt_bbox[:, 1] - anchor_bbox[:, 1]) / anchor_bbox[:, 3]
    tw = np.log(gt_bbox[:, 2] / anchor_bbox[:, 2])
    th = np.log(gt_bbox[:, 3] / anchor_bbox[:, 3])

    return torch.stack([tx, ty, tw, th]).transpose(1, 0)


def repeat_product(x, y):
    coordinate = np.repeat(x, len(y), axis=0)
    height_width = np.stack((np.tile(y[:, 0], len(x)), np.tile(y[:, 1], len(x))), axis=1)
    return np.stack([coordinate[:, 0], coordinate[:, 1], height_width[:, 0], height_width[:, 1]], axis=1)


def generate_anchor_box(feature_map_size, anchor_ratio=[0.5, 1, 2], anchor_size=[128, 256, 512]):
    # generate anchor center coordinate (x,y) for each cell in the feature map
    x = np.arange(8, feature_map_size[0] * 32, 32)
    y = np.arange(8, feature_map_size[1] * 32, 32)
    x, y = np.meshgrid(x, y)
    anchor_coordinate = np.stack((x.ravel(), y.ravel()), axis=1)

    # generate anchor height and width for each cell in the feature map
    anchor_w_h = np.zeros((len(anchor_ratio) * len(anchor_size), 2), dtype=np.float32)
    for i in range(len(anchor_size)):
        for j in range(len(anchor_ratio)):
            index = i * len(anchor_ratio) + j
            anchor_w_h[index, 0] = np.sqrt(anchor_size[i] ** 2 / anchor_ratio[j])
            anchor_w_h[index, 1] = anchor_w_h[index, 0] * anchor_ratio[j]

    anchor_list = repeat_product(anchor_coordinate, anchor_w_h)

    return torch.from_numpy(anchor_list)


def compute_iou(anchor_bbox, gt_bbox):
    '''
    :param anchor_bbox, gt_bbox: be of a set of bbox of the form x,y,w,h
    :return: the iou of an anchor bbox and a grount-truth bbox
    '''
    batch_size = anchor_bbox.shape[0]
    num_gt = gt_bbox.shape[0]

    # convert x,y,w,h to x_min,y_min,x_max,y_max
    convert_anchor_bbox = torch.zeros(anchor_bbox.shape)
    convert_gt_box = torch.zeros(gt_bbox.shape)

    convert_anchor_bbox[:, :2] = anchor_bbox[:, :2] - anchor_bbox[:, 2:] / 2
    convert_anchor_bbox[:, 2:] = anchor_bbox[:, :2] + anchor_bbox[:, 2:] / 2
    convert_gt_box[:, :2] = gt_bbox[:, :2] - gt_bbox[:, 2:] / 2
    convert_gt_box[:, 2:] = gt_bbox[:, :2] + gt_bbox[:, 2:] / 2

    max_xy = torch.min(convert_anchor_bbox[:, 2:].unsqueeze(1).expand(batch_size, num_gt, 2),
                       convert_gt_box[:, 2:].unsqueeze(0).expand(batch_size, num_gt, 2))
    min_xy = torch.max(convert_anchor_bbox[:, :2].unsqueeze(1).expand(batch_size, num_gt, 2),
                       convert_gt_box[:, :2].unsqueeze(0).expand(batch_size, num_gt, 2))
    interact = torch.clamp((max_xy - min_xy), min=0)
    interact = interact[:, :, 0] * interact[:, :, 1]

    area_anchor = ((convert_anchor_bbox[:, 2]-convert_anchor_bbox[:, 0]) * (convert_anchor_bbox[:, 3]-convert_anchor_bbox[:, 1])).unsqueeze(1).expand_as(interact)
    area_gt = ((convert_gt_box[:, 2]-convert_gt_box[:, 0]) * (convert_gt_box[:, 3]-convert_gt_box[:, 1])).unsqueeze(0).expand_as(interact)

    union = area_anchor.float() + area_gt - interact
    return interact / union


def label_anchor(output_bbox, gt_bbox, gt_class, batch_size, is_train):
    anchor_label = []
    anchor_match = []

    for i in range(batch_size):
        tem_label = []
        tem_match = []
        net_output_anchor_bbox = convert_rpn_output_predict_bbox(output_bbox[i])
        iou_list = compute_iou(net_output_anchor_bbox, gt_bbox[i])

        for iou in iou_list:
            if torch.any(iou > 0.3):
                # anchor_label.append(int(gt_class[index][i]))
                tem_label.append(1)
                tem_match.append(iou.argmax())
            elif torch.all(iou < 0.1):
                tem_label.append(0)
                tem_match.append(iou.argmax())
            else:
                tem_label.append(-1)
                tem_match.append(-1)

        tem_label = torch.tensor(tem_label)
        tem_match = torch.tensor(tem_match)

        if is_train:
            low_bound = net_output_anchor_bbox[:, :2] - net_output_anchor_bbox[:, 2:] / 2
            del_index_1 = torch.where(low_bound < 0)[0]

            high_bound = net_output_anchor_bbox[:, :2] + net_output_anchor_bbox[:, 2:] / 2
            del_index_2 = torch.where(high_bound > 800)[0]
        del_index = torch.cat([del_index_1, del_index_2])

        tem_label[del_index] = -1
        tem_match[del_index] = -1

        anchor_label.append(tem_label)
        anchor_match.append(tem_match)

    anchor_label = torch.stack(anchor_label).transpose(1, 0)
    anchor_match = torch.stack(anchor_match).transpose(1, 0)
    # net_output_anchor_bbox = np.delete(net_output_anchor_bbox, delete_list, axis=0)

    return anchor_label, anchor_match


if __name__ == '__main__':
    anchor_list = torch.from_numpy(generate_anchor_box((25, 25), True))
    gt_bbox = [torch.Tensor([[144.3750, 107.8125,  83.7500,  46.4062], [146.8750,  80.1562, 273.7500, 134.5312]]),
               torch.Tensor([[100.3750, 107.8125,  83.7500,  46.4062], [126.8750,  80.1562, 273.7500, 134.5312]])]
    # gt_bbox = [torch.Tensor([[144.3750, 107.8125,  83.7500,  46.4062], [146.8750,  80.1562, 273.7500, 134.5312]])]
    for i in range(2):
        iou_list = compute_iou(anchor_list, torch.stack(gt_bbox)[:, i, :])
        for j, iou in enumerate(iou_list):
            np_iou = iou.numpy()
            if np.any(np_iou > 0.7):
                index = np_iou.argmax()

