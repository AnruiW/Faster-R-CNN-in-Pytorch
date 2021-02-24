import numpy as np


def compute_gt_output(anchor_box, gt_box):
    '''
    :param anchor_box: in the form of x,y,w,h
    :param gt_box: coordinate and WH for a ground-truth box, in the form of x,y,w,h
    :return: compute the ground-truth for the output of the bbox regression
    '''
    tx = (gt_box[0] - anchor_box[0]) / anchor_box[2]
    ty = (gt_box[1] - anchor_box[1]) / anchor_box[3]
    tw = np.log(gt_box[2] / anchor_box[2])
    th = np.log(gt_box[3] / anchor_box[3])

    return np.array([tx, ty, tw, th])


def compute_predict_bbox(rpn_output, anchor_box):
    '''
    :param rpn_output: the output of rpn location output
    :param anchor_box: corresponding anchor box
    :return: convert rpn location output to bbox coordinate
    '''
    x = rpn_output[0] * anchor_box[2] + anchor_box[0]
    y = rpn_output[1] * anchor_box[3] + anchor_box[1]
    w = np.exp(rpn_output[2]) * anchor_box[2]
    h = np.exp(rpn_output[3]) * anchor_box[3]

    return np.array([x, y, w, h])


def repeat_product(x, y):
    coordinate = np.repeat(x, len(y), axis=0)
    height_width = np.stack((np.tile(y[:, 0], len(x)), np.tile(y[:, 1], len(x))), axis=1)
    return np.stack([coordinate[:, 0], coordinate[:, 1], height_width[:, 0], height_width[:, 1]], axis=1)


def generate_anchor_box(feature_map_size, anchor_ratio=[0.5, 1, 2], anchor_size=[128, 256, 512]):
    # generate anchor center coordinate (x,y) for each cell in the feature map
    x = np.arange(8, feature_map_size[0] * 16, 16)
    y = np.arange(8, feature_map_size[1] * 16, 16)
    x, y = np.meshgrid(x, y)
    anchor_coordinate = np.stack((x.ravel(), y.ravel()), axis=1)

    # generate anchor hight and width for each cell in the feature map
    anchor_w_h = np.zeros((len(anchor_ratio) * len(anchor_size), 2), dtype=np.float32)
    for i in range(len(anchor_size)):
        for j in range(len(anchor_ratio)):
            index = i * len(anchor_ratio) + j
            anchor_w_h[index, 0] = np.sqrt(anchor_size[i] ** 2 / anchor_ratio[j])
            anchor_w_h[index, 1] = anchor_w_h[index, 0] * anchor_ratio[j]

    return repeat_product(anchor_coordinate, anchor_w_h)


