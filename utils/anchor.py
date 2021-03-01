import torch
import cv2
import numpy as np


def compute_rpn_gt_output(anchor_bbox, gt_box):
    '''
    :param anchor_bbox: in the form of x,y,w,h
    :param gt_box: coordinate and WH for a ground-truth box, in the form of x,y,w,h
    :return: compute the ground-truth for the output of the bbox regression
    '''
    tx = (gt_box[0] - anchor_bbox[0]) / anchor_bbox[2]
    ty = (gt_box[1] - anchor_bbox[1]) / anchor_bbox[3]
    tw = np.log(gt_box[2] / anchor_bbox[2])
    th = np.log(gt_box[3] / anchor_bbox[3])

    return np.array([tx, ty, tw, th])


def convert_rpn_output_predict_bbox(rpn_output, anchor_bbox):
    '''
    :param rpn_output: the output of rpn location output
    :param anchor_bbox: corresponding anchor box
    :return: convert rpn location output to bbox coordinate
    '''
    x = rpn_output[0] * anchor_bbox[2] + anchor_bbox[0]
    y = rpn_output[1] * anchor_bbox[3] + anchor_bbox[1]
    w = np.exp(rpn_output[2]) * anchor_bbox[2]
    h = np.exp(rpn_output[3]) * anchor_bbox[3]

    return np.array([x, y, w, h])


def compute_gt_output(anchor_bbox, gt_bbox):
    '''
    :param anchor_bbox: in the form of x,y,w,h
    :param gt_bbox: coordinate and WH for a ground-truth box, in the form of x,y,w,h
    :return: the ground-truth for the output of RPN network
    '''
    tx = (gt_bbox[0] - anchor_bbox[0]) / anchor_bbox[2]
    ty = (gt_bbox[1] - anchor_bbox[1]) / anchor_bbox[3]
    tw = np.log(gt_bbox[2] / anchor_bbox[2])
    th = np.log(gt_bbox[3] / anchor_bbox[3])

    return np.array([tx, ty, tw, th])


def repeat_product(x, y):
    coordinate = np.repeat(x, len(y), axis=0)
    height_width = np.stack((np.tile(y[:, 0], len(x)), np.tile(y[:, 1], len(x))), axis=1)
    return np.stack([coordinate[:, 0], coordinate[:, 1], height_width[:, 0], height_width[:, 1]], axis=1)


def generate_anchor_box(feature_map_size, is_train, anchor_ratio=[0.5, 1, 2], anchor_size=[128, 256, 512]):
    # generate anchor center coordinate (x,y) for each cell in the feature map
    x = np.arange(8, feature_map_size[0] * 32, 32)
    y = np.arange(8, feature_map_size[1] * 32, 32)
    x, y = np.meshgrid(x, y)
    anchor_coordinate = np.stack((x.ravel(), y.ravel()), axis=1)

    # generate anchor hight and width for each cell in the feature map
    anchor_w_h = np.zeros((len(anchor_ratio) * len(anchor_size), 2), dtype=np.float32)
    for i in range(len(anchor_size)):
        for j in range(len(anchor_ratio)):
            index = i * len(anchor_ratio) + j
            anchor_w_h[index, 0] = np.sqrt(anchor_size[i] ** 2 / anchor_ratio[j])
            anchor_w_h[index, 1] = anchor_w_h[index, 0] * anchor_ratio[j]

    anchor_list = repeat_product(anchor_coordinate, anchor_w_h)

    if is_train:
        low_bound = anchor_list[:, :2] - anchor_list[:, 2:] / 2
        del_index = np.where(low_bound < 0)[0]
        anchor_list = np.delete(anchor_list, del_index, axis=0)

        high_bound = anchor_list[:, :2] + anchor_list[:, 2:] / 2
        del_index = np.where(high_bound > 800)[0]
        anchor_list = np.delete(anchor_list, del_index, axis=0)

    return anchor_list


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


def label_anchor(anchor_bbox, gt_bbox, gt_class, batch_size):
    print(gt_class)
    anchor_label = []
    for i in range(batch_size):
        iou_list = compute_iou(anchor_bbox, torch.stack(gt_bbox)[:, i, :])
        for iou in iou_list:
            print(iou)


    raise Exception


if __name__ == '__main__':
    anchor_list = torch.from_numpy(generate_anchor_box((25, 25), True))
    gt_bbox = [torch.Tensor([[144.3750, 107.8125,  83.7500,  46.4062], [146.8750,  80.1562, 273.7500, 134.5312]]),
               torch.Tensor([[100.3750, 107.8125,  83.7500,  46.4062], [126.8750,  80.1562, 273.7500, 134.5312]])]
    # gt_bbox = [torch.Tensor([[144.3750, 107.8125,  83.7500,  46.4062], [146.8750,  80.1562, 273.7500, 134.5312]])]
    for i in range(2):
        iou_list = compute_iou(anchor_list, torch.stack(gt_bbox)[:, i, :])
        print(iou_list)
    raise Exception
