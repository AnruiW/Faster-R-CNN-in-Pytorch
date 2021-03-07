import torch
import time
import os
from torch import optim
from torch.nn import SmoothL1Loss, CrossEntropyLoss
from Object_Detection.Faster_RCNN_in_Pytorch.utils.anchor import compute_iou, generate_anchor_box, label_anchor, compute_rpn_gt_output


def rpn_loss(proposal_list, score_list, anchor_label, anchor_match, gt_bbox, gt_class):
    '''
    :param proposal_list: the original output of the RPN network
    :param score_list: the original output of the RPN network
    :param gt_bbox: convert the
    :param gt_score:
    '''
    anchor_list = generate_anchor_box((25, 25))

    expand_gt_bbox = []
    expand_gt_score = []
    for i in range(len(proposal_list)):
        tem_gt_bbox = []
        tem_gt_score = []
        for j, match in enumerate(anchor_match[:, i]):
            if match != -1:
                tem_gt_bbox.append(proposal_list[i][j])
                tem_gt_score.append(0)
            else:
                tem_gt_bbox.append(compute_rpn_gt_output(anchor_list[j], gt_bbox[i][match]))
                tem_gt_score.append(1)

        expand_gt_bbox.append(torch.stack(tem_gt_bbox))
        expand_gt_score.append(torch.tensor(tem_gt_score))
    expand_gt_bbox = torch.stack(expand_gt_bbox)
    expand_gt_score = torch.stack(expand_gt_score)

    location_loss = SmoothL1Loss()
    l_loss = []
    for i in range(len(proposal_list)):
        tem_l_loss = 0
        for j in range(len(expand_gt_bbox[i])):
            tem_l_loss += expand_gt_score[i][j] * location_loss(expand_gt_bbox[i][j], proposal_list[i][j])
        l_loss.append(tem_l_loss * 10 / len(proposal_list[0]))

    classification_loss = CrossEntropyLoss()
    s_loss = []
    for i in range(len(proposal_list)):
        tem_s_loss = 0
        for j in range(len(score_list[i])):
            tem_s_loss += classification_loss(score_list[i][j].unsqueeze(0), expand_gt_score[i][j].unsqueeze(0))
        s_loss.append(tem_s_loss / len(proposal_list))

    return (torch.stack(l_loss) + torch.stack(s_loss)).sum()


def frcnn_loss(proposal_list, score_list, gt_bbox, gt_class):
    location_loss = SmoothL1Loss()
    classification_loss = CrossEntropyLoss()

    l_loss = 0
    s_loss = 0
    for i in range(len(proposal_list)):
        iou_list = compute_iou(proposal_list[i], gt_bbox[:, i, :])
        predi_label_list = score_list[i].argmax(dim=1)
        batch_label_list = iou_list.argmax(dim=1)

        gt_bbox_list = []
        gt_class_list = []

        for j, index in enumerate(batch_label_list):
            if predi_label_list[j] == 0:
                gt_bbox_list.append(proposal_list[j])
            else:
                gt_bbox_list.append(gt_bbox[:, i, :][index])
            gt_class_list.append(gt_class[index][i])

        gt_bbox_list = torch.stack(gt_bbox_list)
        gt_class_list = torch.stack(gt_class_list)
        l_loss += location_loss(proposal_list[i], gt_bbox_list)
        s_loss += classification_loss(score_list[i], gt_class_list)

    return l_loss + s_loss


def evaluate_val_accuracy(net, test_iter, device):
    net.eval()
    num_image, accuracy = 0, 0

    for test_image, test_label in test_iter:
        with torch.no_grad:
            accuracy += (net(test_image.to(device)).argmax(dim=1) == test_label.to(device)).float().sum().cpu().item()
        num_image += len(test_image)

    net.train()
    return accuracy / num_image


def evaluate_rpn_accuracy(net, test_iter, device):
    net.eval()
    num_image, accuracy, iou = 0, 0, 0

    for test_image, test_label in test_iter:
        with torch.no_grad:
            # compute iou to measure accuracy
            output_list, _ = net(test_image.to(device))
            iou_list = compute_iou(output_list, test_label)


def train_net(net, net_name, train_iter, test_iter, optimizer, lr_scheduler, num_epoch, device):
    net = net.to(device)
    print(f"*************training {net_name} net on {device}*************")
    checkpoint_dir = os.path.join(os.path.abspath('.'), f'{net_name}_checkpoint.pth')
    start_time = time.time()

    if os.path.exists(os.path.join(os.path.abspath('.'), f'{net_name}_final.pth')):
        net.load_state_dict(torch.load(f'{net_name}_final.pth'))

    for epoch in range(num_epoch):
        epoch_loss = 0
        for image_list, label_list in train_iter:
            image_list = image_list.to(device)

            if net_name == 'VGG':
                label_list.to(device)
                output_list = net(image_list)
                criteria = torch.nn.CrossEntropyLoss()
                loss = criteria(output_list, label_list)

            elif net_name == 'RPN':
                _, proposal_list, score_list = net(image_list)
                # print(label_list)
                gt_class = label_list[0]
                gt_bbox = label_list[1]

                anchor_label, anchor_match = label_anchor(proposal_list, gt_bbox, gt_class, image_list.shape[0], True)
                loss = rpn_loss(proposal_list, score_list, anchor_label, anchor_match, gt_bbox, gt_class)

            elif net_name == 'Faster R CNN':
                proposal_list, score_list = net(image_list)
                gt_class = label_list[0]
                gt_bbox = torch.cat([i.unsqueeze(0) for i in label_list[1]], dim=0)
                loss = frcnn_loss(proposal_list, score_list, gt_bbox, gt_class)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step(loss)

            current_epoch_loss = loss.cpu().item()
            epoch_loss += current_epoch_loss

            print(f"RPN: batch training loss: {current_epoch_loss}, batch training time: {time.time()-start_time}")

        # if net_name == 'VGG':
        #     val_accuracy = evaluate_val_accuracy(net, test_iter, device)
        # elif net_name == 'RPN':
        #     val_accuracy = 0
        # elif net_name == 'Faster R CNN':
        #     val_accuracy = 0

        print(f"RPN: epoch: {epoch+1}, epoch training loss: {epoch_loss}, epoch training time: {time.time()-start_time}")
        start_time = time.time()

        torch.save(net, checkpoint_dir)
        print(f"saving model to {checkpoint_dir}")

    torch.save(net, os.path.join(os.path.abspath(), f'{net_name}_final.pth'))
    print(f"saving final model to {os.path.join(os.path.abspath(), f'{net_name}_final.pth')}")


def train_fasterrcnn(frcnn, rpn, rpn_lr, frcnn_lr, train_iter, test_iter, device):
    # 4 step alternating training
    rpn_optimizer = optim.SGD(rpn.parameters(), rpn_lr, momentum=0.9, weight_decay=5e-4)
    lr_scheduler = optim.lr_scheduler.StepLR(rpn_optimizer, step_size=30, gamma=0.1)

    # 第一轮rpn跟Faster rcnn不应该共享参数
    train_net(rpn, 'RPN', train_iter, test_iter, rpn_optimizer, lr_scheduler, 45, device)
    train_net(frcnn, 'Faster R CNN', train_iter, test_iter, rpn_optimizer, lr_scheduler, 45, device)
    train_net(rpn, 'RPN', train_iter, test_iter, rpn_optimizer, lr_scheduler, 45, device)
    train_net(frcnn, 'Faster R CNN', train_iter, test_iter, rpn_optimizer, lr_scheduler, 45, device)



