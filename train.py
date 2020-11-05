from __future__ import print_function
import os

from focal_loss import FocalLoss
from test import *
from dataset import *
import torch
from torch.utils import data
import torch.nn.functional as F
from models import *
import torchvision
from sklearn import preprocessing

from visualizer import *
from view_model import *
import torch
import numpy as np
import random
import time
from config import *
from torch.nn import DataParallel
from torchvision import datasets
from torch.optim.lr_scheduler import StepLR
from test import *
from resnet import *
from metrics import *
from torchvision import datasets
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler


def save_model(model, save_path, name, iter_cnt):
    ensure_dir(save_path)
    save_name = os.path.join(save_path, name + '_' + str(iter_cnt) + '.pth')
    torch.save(model.state_dict(), save_name)
    return save_name

def ensure_dir(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)


if __name__ == '__main__':

    opt = Config()
    # python -m visdom.server, http://localhost:8097
    if opt.display:
        visualizer = Visualizer()
    device = torch.device("cuda")

    dataset = Dataset2(opt.train_root, opt.total_dataset_list,input_shape=opt.input_shape)
    # dataloader = data.DataLoader(dataset,
    #                               batch_size=opt.train_batch_size,
    #                               shuffle=True,
    #                               num_workers=opt.num_workers)

    validation_split = 0.2

    dataset_len = len(dataset)
    indices = list(range(dataset_len))

    # Randomly splitting indices:
    val_len = int(np.floor(validation_split * dataset_len))
    validation_idx = np.random.choice(indices, size=val_len, replace=False)
    train_idx = list(set(indices) - set(validation_idx))

    ## Defining the samplers for each phase based on the random indices:
    train_sampler = SubsetRandomSampler(train_idx)
    validation_sampler = SubsetRandomSampler(validation_idx)

    # These loops write the randomly sampeled datasets into txt files and save them in facebank directory
    os.chdir(r'C:\Users\noamb\PycharmProjects\Volcani\arcface-pytorch\data\facebank')
    train_file = open('sampeled_train.txt', 'w')
    for i_train in range(len(train_sampler.indices)):
        train_file.write( '%s'%dataset.imgs[train_sampler.indices[i_train]] + '\n' )
    train_file.close()

    val_file = open('sampeled_val.txt', 'w')
    for i_val in range(len(validation_sampler.indices)):
        val_file.write('%s' % dataset.imgs[validation_sampler.indices[i_val]] + '\n')
    val_file.close()

    # recreates datasets and Dataloaders with the original implementation 'phase' feature
    train_dataset = Dataset(opt.train_root, opt.train_list,input_shape=opt.input_shape,phase='train')
    val_dataset = Dataset(opt.train_root, opt.val_list,input_shape=opt.input_shape,phase='val')

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                              batch_size=opt.train_batch_size,
                                              shuffle=True,
                                              num_workers=opt.num_workers)
    validation_loader = torch.utils.data.DataLoader(val_dataset,
                                                batch_size=opt.validation_batch_size,
                                                shuffle=True,
                                                num_workers=opt.num_workers)

    data_loaders = {"train": train_loader, "val": validation_loader}
    data_lengths = {"train": len(train_idx), "val": val_len}

    # train_dataset = Dataset(opt.train_root, opt.train_list, phase='train', input_shape=opt.input_shape)
    # trainloader = data.DataLoader(train_dataset,
    #                               batch_size=opt.train_batch_size,
    #                               shuffle=True,
    #                               num_workers=opt.num_workers)
    #
    # validation_dataset = Dataset(opt.train_root, opt.val_list, phase='train', input_shape=opt.input_shape)
    # validationloader = data.DataLoader(train_dataset,
    #                               batch_size=opt.test_batch_size,
    #                               shuffle=True,
    #                               num_workers=opt.num_workers)

    finding_to_label = { '3401': 0, '3405': 1, '3429': 2, '3505': 3, '3576': 4, '3596': 5,
                        '3628': 6, '3634': 7, '3677': 8, '3689': 9, '3698': 10, '3713': 11,
                        '3721': 12, '3724': 13, '3732': 14, '3740': 15, '3758': 16, '3812': 17,
                        '3813': 18, '3814': 19, '3855': 20, '3871': 21, '3889': 22,
                        '3893': 23, '3898': 24, '3905': 25, '3912': 26, '3914': 27, '3916': 28,
                        '3918': 29, '3931': 30, '3941': 31, '3942': 32, '3943': 33, '3947': 34,
                        '3953': 35, '3955': 36, '3970': 37, '7836': 38, '7842': 39, '7858': 40,
                        '7895': 41  }
    # label_to_finding = {v: k for k, v in finding_to_label.items()}

    #label_encoder = preprocessing.LabelEncoder()
    print('{} train iters per epoch:'.format(len(train_loader)))

    if opt.loss == 'focal_loss':
        criterion = FocalLoss(gamma=2)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    if opt.backbone == 'resnet18':
        model = resnet_face18(use_se=opt.use_se)
    elif opt.backbone == 'resnet34':
        model = resnet34()
    elif opt.backbone == 'resnet50':
        model = resnet50()

    if opt.metric == 'add_margin':
        metric_fc = AddMarginProduct(512, opt.num_classes, s=30, m=0.35)
    elif opt.metric == 'arc_margin':
        metric_fc = ArcMarginProduct(512, opt.num_classes, s=30, m=0.5, easy_margin=opt.easy_margin)
    elif opt.metric == 'sphere':
        metric_fc = SphereProduct(512, opt.num_classes, m=4)
    else:
        metric_fc = nn.Linear(512, opt.num_classes)

    # view_model(model, opt.input_shape)
    print(model)
    model.to(device)
    model = DataParallel(model)
    metric_fc.to(device)
    metric_fc = DataParallel(metric_fc)

    if opt.optimizer == 'sgd':
        optimizer = torch.optim.SGD([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                    lr=opt.lr, weight_decay=opt.weight_decay)
    else:
        optimizer = torch.optim.Adam([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                     lr=opt.lr, weight_decay=opt.weight_decay)
    scheduler = StepLR(optimizer, step_size=opt.lr_step, gamma=0.1)

    start = time.time()
#     for i in range(opt.max_epoch):
#         scheduler.step()
#         model.train()
#         for ii, data in enumerate(train_loader):
#             data_input, finding = data
#             # Convert finding to classifier label
#             finding_arr = finding.numpy()
#             labels = []
#             for v in finding_arr:
#                 labels.append(finding_to_label['%d'%v] )
#
#             label = torch.IntTensor(labels)
#             data_input = data_input.to(device)
#             label = label.to(device).long()
#             feature = model(data_input)
#             output = metric_fc(feature, label)
#             loss = criterion(output, label)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#             iters = i * len(train_loader) + ii
#
#             if iters % opt.print_freq == 0:
#                 output = output.data.cpu().numpy()
#                 output = np.argmax(output, axis=1)
#                 label = label.data.cpu().numpy()
#                 # print(output)
#                 # print(label)
#                 acc = np.mean((output == label).astype(int))
#                 speed = opt.print_freq / (time.time() - start)
#                 time_str = time.asctime(time.localtime(time.time()))
#                 print('{} train epoch {} iter {} {} iters/s loss {} acc {}'.format(time_str, i, ii, speed, loss.item(), acc))
#                 if opt.display:
#                     visualizer.display_current_results(iters, loss.item(), name='train_loss')
#                     visualizer.display_current_results(iters, acc, name='train_acc')
#
#                 start = time.time()
#
#         if i % opt.save_interval == 0 or i == opt.max_epoch:
#             save_model(model, opt.checkpoints_path, opt.backbone, i)
#
#         model.eval()
#         acc = lfw_test(model, img_paths, identity_list, opt.val_list, opt.test_batch_size)
#         if opt.display:
#             visualizer.display_current_results(iters, acc, name='test_acc')
#
# #################
    for epoch in range(opt.max_epoch):
        scheduler.step()
        print('Epoch {}/{}'.format(epoch, opt.max_epoch - 1))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                #optimizer = scheduler(optimizer, epoch)
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode
            for ii, data in enumerate(data_loaders[phase]):
                data_input, finding = data
                # Convert finding to classifier label
                finding_arr = finding.numpy()
                labels = []
                for v in finding_arr:
                    labels.append(finding_to_label['%d' % v])
                label = torch.IntTensor(labels)
                data_input = data_input.to(device)
                label = label.to(device).long()
                feature = model(data_input)
                output = metric_fc(feature, label)
                loss = criterion(output, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                iters = epoch * len(data_loaders[phase]) + ii
                if iters % opt.print_freq == 0:
                    output = output.data.cpu().numpy()
                    output = np.argmax(output, axis=1)
                    label = label.data.cpu().numpy()
                    # print(output)
                    # print(label)
                    acc = np.mean((output == label).astype(int))
                    speed = opt.print_freq / (time.time() - start)
                    time_str = time.asctime(time.localtime(time.time()))
                    if phase == 'train':
                        print('{} train epoch {} iter {} {} iters/s loss {} acc {}'.format(time_str, epoch, ii, speed,
                                                                                            loss.item(), acc) )
                    else:
                        print('{} validation epoch {} iter {} {} iters/s loss {} acc {}'.format(time_str, epoch, ii, speed,
                                                                                           loss.item(), acc))
                    if opt.display:
                        if phase == 'train':
                            visualizer.display_current_results(iters, loss.item(), name='train_loss')
                            visualizer.display_current_results(iters, acc, name='train_acc')
                        else:
                            visualizer.display_current_results(iters, loss.item(), name='validation_loss')
                            visualizer.display_current_results(iters, acc, name='validation_acc')

                    start = time.time()

            if epoch % opt.save_interval == 0 or epoch == opt.max_epoch:
                save_model(model, opt.checkpoints_path, opt.backbone, epoch)



            # running_loss = 0.0
            #
            # # Iterate over data.
            # for data in data_loaders[phase]:
            #
            #     # get the input images and their corresponding labels
            #     images = data['image']
            #     key_pts = data['keypoints']
            #
            #     # flatten pts
            #     key_pts = key_pts.view(key_pts.size(0), -1)
            #
            #     # wrap them in a torch Variable
            #     images, key_pts = Variable(images), Variable(key_pts)
            #
            #     # convert variables to floats for regression loss
            #     key_pts = key_pts.type(torch.FloatTensor)
            #     images = images.type(torch.FloatTensor)
            #
            #     # forward pass to get outputs
            #     output_pts = net(images)
            #
            #     # calculate the loss between predicted and target keypoints
            #     loss = criterion(output_pts, key_pts)
            #
            #     # zero the parameter (weight) gradients
            #     optimizer.zero_grad()
            #
            #     # backward + optimize only if in training phase
            #     if phase == 'train':
            #         loss.backward()
            #         # update the weights
            #         optimizer.step()
            #
            #     # print loss statistics
            #     running_loss += loss.data[0]
            #
            # epoch_loss = running_loss / data_lengths[phase]
            # print('{} Loss: {:.4f}'.format(phase, epoch_loss))