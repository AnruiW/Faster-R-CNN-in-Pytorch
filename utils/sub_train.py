import torch
import time
import os
from torch import optim
from torch.nn import SmoothL1Loss, CrossEntropyLoss

# 可能输出很多proposal
# 输入就是一个y要手动分
def rpn_loss(proposal_list, score_list, gt_bbox, gt_score):
    '''
    :param proposal_list: the original output of the RPN network
    :param score_list: the original output of the RPN network
    :param gt_bbox: convert the
    :param gt_score:
    :return:
    '''
    location_loss = SmoothL1Loss()
    classification_loss = CrossEntropyLoss()
    l_loss = location_loss(proposal_list, gt_bbox)
    s_loss = classification_loss(score_list, gt_score)
    return l_loss + gt_score * s_loss


def frcnn_loss(proposal_list, score_list, gt_bbox, gt_score):



def evaluate_val_accuracy(net, test_iter, device):
    net.eval()
    num_image, accuracy = 0, 0

    for test_image, test_label in test_iter:
        with torch.no_grad:
            accuracy += (net(test_image.to(device)).argmax(dim=1) == test_label.to(device)).float().sum().cpu().item()
        num_image += len(test_image)

    net.train()
    return accuracy / num_image


def train_vgg(vgg, train_iter, test_iter, optimizer, lr_scheduler, num_epoch, device):
    vgg = vgg.to(device)
    print("* * * * * * *training VGG net on {}* * * * * * *".format(device))
    checkpoint_dir = os.path.join(os.path.abspath('.'), 'vgg_checkpoint.pth')

    criteria = torch.nn.CrossEntropyLoss()

    if os.path.exists(checkpoint_dir):
        vgg.load_state_dict(torch.load(checkpoint_dir))
        print("successfully load VGG model parameters")

    start_time = time.time()

    for epoch in range(num_epoch):
        epoch_loss = 0
        for image_list, label_list in train_iter:
            image_list.to(device)
            label_list.to(device)

            output_list = vgg(image_list)
            loss = criteria(output_list, label_list)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step(loss)

            current_epoch_loss = loss.cpu().item()
            epoch_loss += current_epoch_loss

            print(f"VGG: batch training loss: {current_epoch_loss}, batch training time: {time.time()-start_time}")

        val_accuracy = evaluate_val_accuracy(vgg, test_iter, device)
        print(f"VGG: epoch: {epoch+1}, epoch training loss: {epoch_loss}, validation accuracy: {val_accuracy}, epoch training time: {time.time()-start_time}")
        start_time = time.time()

        torch.save(vgg, checkpoint_dir)
        print(f"saving model to {checkpoint_dir}")

    torch.save(vgg, os.path.join(os.path.abspath(), 'vgg_checkpoint.pth'))
    print(f"saving final model to {os.path.join(os.path.abspath(), 'vgg_final.pth')}")


def train_rpn(rpn, train_iter, test_iter, optimizer, lr_scheduler, num_epoch, device):
    rpn = rpn.to(device)
    print("* * * * * * *training RPN net on {}* * * * * * *".format(device))
    checkpoint_dir = os.path.join(os.path.abspath('.'), 'rpn_checkpoint.pth')

    criteria = torch.aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa

    if os.path.exists(checkpoint_dir):
        rpn.load_state_dict(torch.load(checkpoint_dir))
        print("successfully load RPN model parameters")

    start_time = time.time()

    for epoch in range(num_epoch):
        epoch_loss = 0
        for image_list, label_list in train_iter:
            image_list.to(device)
            label_list.to(device)

            proposal_list, score_list = rpn(image_list)
            loss = criteria(proposal_list, score_list)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step(loss)

            current_epoch_loss = loss.cpu().item()
            epoch_loss += current_epoch_loss

            print(f"RPN: batch training loss: {current_epoch_loss}, batch training time: {time.time()-start_time}")

        val_accuracy = evaluate_val_accuracy(rpn, test_iter, device)
        print(f"RPN: epoch: {epoch+1}, epoch training loss: {epoch_loss}, validation accuracy: {val_accuracy}, epoch training time: {time.time()-start_time}")
        start_time = time.time()

        torch.save(rpn, checkpoint_dir)
        print(f"saving model to {checkpoint_dir}")

    torch.save(rpn, os.path.join(os.path.abspath(), 'rpn_final.pth'))
    print(f"saving final model to {os.path.join(os.path.abspath(), 'rpn_final.pth')}")


def train_frcnn(frcnn, train_iter, test_iter, optimizer, lr_scheduler, num_epoch, device):
    frcnn = frcnn.to(device)
    print("* * * * * * *training Faster R CNN net on {}* * * * * * *".format(device))
    checkpoint_dir = os.path.join(os.path.abspath('.'), 'frcnn_checkpoint.pth')

    if os.path.exists(checkpoint_dir):
        frcnn.load_state_dict(torch.load(checkpoint_dir))
        print("successfully load Faster R CNN model parameters")

    start_time = time.time()

    for epoch in range(num_epoch):
        epoch_loss = 0
        for image_list, label_list in train_iter:
            image_list.to(device)
            label_list.to(device)

            proposal_list, score_list = frcnn(image_list)
            loss = criteria(proposal_list, score_list)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step(loss)

            current_epoch_loss = loss.cpu().item()
            epoch_loss += current_epoch_loss

            print(f"RPN: batch training loss: {current_epoch_loss}, batch training time: {time.time()-start_time}")

        val_accuracy = evaluate_val_accuracy(frcnn, test_iter, device)
        print(f"RPN: epoch: {epoch+1}, epoch training loss: {epoch_loss}, validation accuracy: {val_accuracy}, epoch training time: {time.time()-start_time}")
        start_time = time.time()

        torch.save(frcnn, checkpoint_dir)
        print(f"saving model to {checkpoint_dir}")

    torch.save(frcnn, os.path.join(os.path.abspath(), 'rpn_final.pth'))
    print(f"saving final model to {os.path.join(os.path.abspath(), 'rpn_final.pth')}")


def train_net(net, net_name, train_iter, test_iter, optimizer, lr_scheduler, num_epoch, device):
    net = net.to(device)
    print(f"*************training {net_name} net on {device}*************")
    checkpoint_dir = os.path.join(os.path.abspath('.'), f'{net_name}_checkpoint.pth')

    # if net_name == 'VGG':
    #     criteria = torch.nn.CrossEntropyLoss()
    # elif net_name == 'RPN':
    #     criteria = rpn_loss
    # elif net_name == 'Faster R CNN':
    #     criteria = frcnn_loss

    for epoch in range(num_epoch):
        epoch_loss = 0
        for image_list, label_list in train_iter:
            image_list.to(device)
            label_list.to(device)

            if net_name == 'VGG':
                output_list = net(image_list)
                criteria = torch.nn.CrossEntropyLoss()
                loss = criteria(output_list, label_list)
            elif net_name == 'RPN':
                proposal_list, score_list = net(image_list)
                gt_bbox =
                gt_score =
                loss = rpn_loss(proposal_list, score_list, gt_bbox, gt_score)
            elif net_name == 'Faster R CNN':







def train_fasterrcnn(frcnn, rpn, rpn_lr, frcnn_lr, device):
    # 4 step alternating training
    rpn_train_set = 0
    rpn_val_set = 0

    rpn_optimizer = optim.SGD(rpn.parameters(), rpn_lr, momentum=0.9, weight_decay=5e-4)
    lr_scheduler = optim.lr_scheduler.StepLR(rpn_optimizer, step_size=30, gamma=0.1)

    train_rpn(rpn, rpn_train_set, rpn_val_set, rpn_optimizer, lr_scheduler, 45, device)






