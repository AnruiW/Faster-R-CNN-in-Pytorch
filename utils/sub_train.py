import torch
import time
import os
from torch import optim
from torch.nn import SmoothL1Loss, CrossEntropyLoss
from Object_Detection.Faster_RCNN_in_Pytorch.utils.anchor import compute_iou, generate_anchor_box, label_anchor


# 可能输出很多proposal
# 输入就是一个y要手动分
def rpn_loss(proposal_list, score_list, anchor_label, anchor_match, gt_bbox, gt_class):
    '''
    :param proposal_list: the original output of the RPN network
    :param score_list: the original output of the RPN network
    :param gt_bbox: convert the
    :param gt_score:
    '''
    print()
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++")

    print(proposal_list)
    print(gt_bbox)
    print()
    print(anchor_label)
    print(anchor_match)
    print(torch.where(anchor_match != -1))
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
                tem_gt_bbox.append(gt_bbox[match][i])
                tem_gt_score.append(1)
        expand_gt_bbox.append(torch.stack(tem_gt_bbox))
        expand_gt_score.append(torch.tensor(tem_gt_score))
    expand_gt_bbox = torch.stack(expand_gt_bbox)
    expand_gt_score = torch.stack(expand_gt_score)
    print(proposal_list.shape)
    print(expand_gt_bbox.shape)
    location_loss = SmoothL1Loss()
    l_loss = location_loss(proposal_list, expand_gt_bbox)


    print(score_list.shape)
    print(expand_gt_score.shape)

    classification_loss = CrossEntropyLoss(weight=torch.tensor([0, 1]))

    s_loss = []

    for i in range(len(proposal_list)):
        tem_s_loss = 0
        for j in range(len(score_list[i])):
            print(expand_gt_bbox[i])
            print(score_list[i][j])
            tem_s_loss += expand_gt_bbox[i] * classification_loss(score_list[i][j], expand_gt_bbox[i])
        s_loss.append(tem_s_loss)



    # classification_loss(score_list, expand_gt_score)
    print(l_loss)
    print(s_loss)
    return l_loss + gt_class * s_loss


def frcnn_loss(proposal_list, score_list, gt_bbox, gt_score):
    location_loss = SmoothL1Loss()
    classification_loss = CrossEntropyLoss()
    l_loss = location_loss(proposal_list, gt_bbox)
    s_loss = classification_loss(score_list, gt_score)




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

    for epoch in range(num_epoch):
        epoch_loss = 0
        for image_list, label_list in train_iter:
            image_list.to(device)

            if net_name == 'VGG':
                label_list.to(device)
                output_list = net(image_list)
                criteria = torch.nn.CrossEntropyLoss()
                loss = criteria(output_list, label_list)
            elif net_name == 'RPN':
                proposal_list, score_list = net(image_list)
                print(proposal_list)
                print(score_list)
                print(label_list)
                gt_class = label_list[0]
                gt_bbox = label_list[1]


                # anchor_list = generate_anchor_box((image_list.shape[2], image_list.shape[3]))
                anchor_label, anchor_match = label_anchor(proposal_list, gt_bbox, gt_class, image_list.shape[0], True)

                loss = rpn_loss(proposal_list, score_list, anchor_label, anchor_match, gt_bbox, gt_class)
            elif net_name == 'Faster R CNN':
                pass

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step(loss)

            current_epoch_loss = loss.cpu().item()
            epoch_loss += current_epoch_loss

            print(f"RPN: batch training loss: {current_epoch_loss}, batch training time: {time.time()-start_time}")

        if net_name == 'VGG':
            val_accuracy = evaluate_val_accuracy(net, test_iter, device)
        elif net_name == 'RPN':
            val_accuracy = 0
        elif net_name == 'Faster R CNN':
            val_accuracy = 0

        print(f"RPN: epoch: {epoch+1}, epoch training loss: {epoch_loss}, validation accuracy: {val_accuracy}, epoch training time: {time.time()-start_time}")
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






