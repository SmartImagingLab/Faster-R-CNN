# --------------------------------------------------------
# Pytorch FPN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jianwei Yang, based on code from faster R-CNN
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
sys.path.append('..')
import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
import netron
import logging
import pickle as cPickle
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from datetime import datetime

import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from lib.model.utils.scheduler import GradualWarmupScheduler
from lib.roi_data_layer.roidb_v1 import combined_roidb
from lib.roi_data_layer.roibatchLoader import roibatchLoader
from lib.model.utils.net_utils import weights_normal_init, save_net, load_net, \
    adjust_learning_rate, save_checkpoint, clip_gradient
from lib.model.rpn.bbox_transform import bbox_transform_inv
from lib.model.rpn.bbox_transform import clip_boxes
from lib.model.soft_nms.nms import soft_nms
from lib.model.soft_nms.nms import nms
from lib.model.utils.net_utils import vis_detections
from lib.model.fpn.resnet_IN import resnet
# from model.fpn.detnet_backbone import detnet
from lib.model.fpn.normalization import Regularization
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import pdb

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--dataset', dest='dataset',
                        help='training dataset',
                        #default='EP_detect', type=str)   
                        default='SKA_detect', type=str)          # can alter
    parser.add_argument('--net', dest='net',
                        help='res101, res152, res50,etc',
                        default='res50', type=str)                # can alter
    parser.add_argument('--start_epoch', dest='start_epoch',
                        help='starting epoch',
                        default=1, type=int)
    parser.add_argument('--epochs', dest='max_epochs',
                        help='number of epochs to train',
                        default=300, type=int)                       # can alter
    parser.add_argument('--disp_interval', dest='disp_interval',
                        help='number of iterations to display',
                        default=10, type=int)                        # can alter
    parser.add_argument('--checkpoint_interval', dest='checkpoint_interval',
                        help='number of iterations to display',
                        default=100, type=int)            # can alter 10000

    parser.add_argument('--save_dir', dest='save_dir',
                        help='directory to save models', default="../model_save",
                        nargs=argparse.REMAINDER)
    parser.add_argument('--num_workers', dest='num_workers',
                        help='number of worker to load data',
                        default=0, type=int)
    parser.add_argument('--cuda', dest='cuda',
                        help='whether use CUDA',
                        action='store_true',default=True)
    parser.add_argument('--mGPUs', dest='mGPUs',
                        help='whether use multiple GPUs',
                        action='store_true',default=False)
    parser.add_argument('--lscale', dest='lscale',
                        help='whether use large scale',
                        action='store_true')
    parser.add_argument('--train_bs', dest='batch_size',
                        help='batch_size',
                        default=1, type=int)                      # can alter
    parser.add_argument('--test_bs', dest='test_batch_size',
                        help='batch_size',
                        default=1, type=int)
    parser.add_argument('--cag', dest='class_agnostic',
                        help='whether perform class_agnostic bbox regression',
                        action='store_true')
    parser.add_argument('--vis', dest='vis',
                        help='visualization mode',
                        action='store_true')

    # config optimization
    parser.add_argument('--o', dest='optimizer',
                        help='training optimizer',
                        default="adam", type=str)
    parser.add_argument('--lr', dest='lr',
                        help='starting learning rate',
                        default=0.000025, type=float)           # can alter
    parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                        help='step to do learning rate decay, unit is epoch',
                        default=2, type=int)
    parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                        help='learning rate decay ratio',
                        default=0.5, type=float)           # can alter：0.5

    # set training session
    parser.add_argument('--s', dest='session',
                        help='training session',
                        default=333333333333, type=int)                  # can alter

    # resume trained model (加载训练好的模型时用)
    parser.add_argument('--r', dest='resume',
                        help='resume checkpoint or not',
                        default=False, type=bool)
    parser.add_argument('--checksession', dest='checksession',
                        help='checksession to load model',
                        default=6, type=int)                   # can alter
    parser.add_argument('--checkepoch', dest='checkepoch',
                        help='checkepoch to load model',
                        default=50, type=int)                    # can alter
    parser.add_argument('--checkpoint', dest='checkpoint',
                        help='checkpoint to load model',
                        default=125, type=int)                 # can alter
    parser.add_argument('--use_scheduler', dest='use_scheduler',
                        help='whether use scheduler learning rate',
                        default=True, type=bool)
    parser.add_argument('--Use_regularization', dest='Use_regularization',
                        help='whether Use regularization',
                        default=True, type=bool)
    parser.add_argument('--torch1_1', dest='torch1_1',
                        help='whether Use torch1-1',
                        default=True, type=bool)

    parser.add_argument('--channel',dest='channel',
                        help='channel of dataset',
                        default=144,type=int)

    args = parser.parse_args()
    return args


class sampler(Sampler):
    def __init__(self, train_size, batch_size):
        num_data = train_size
        self.train_size = train_size
        self.num_per_batch = int(num_data / batch_size)
        self.batch_size = batch_size
        self.range = torch.arange(0,batch_size).view(1, batch_size).long()
        self.leftover_flag = False
        if num_data % batch_size:
            self.leftover = torch.arange(self.num_per_batch*batch_size, num_data).long()
            self.leftover_flag = True
    def __iter__(self):
        rand_num = torch.randperm(self.num_per_batch).view(-1,1) * self.batch_size
        self.rand_num = rand_num.expand(self.num_per_batch, self.batch_size) + self.range

        self.rand_num_view = self.rand_num.view(-1)

        if self.leftover_flag:
            self.rand_num_view = torch.cat((self.rand_num_view, self.leftover),0)

        return iter(self.rand_num_view)

    def __len__(self):
        return self.train_size

def _print(str, logger):
    print(str)
    logger.info(str)

if __name__ == '__main__':

    args = parse_args()

    if not args.mGPUs:
        os.environ["CUDA_VISIBLE_DEVICES"] = '1'      # select Gpu
        print("GPU {} will be used\n".format('1'))
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # 此时显示2块显卡

    print('Called with args:')
    print(args)

    logging.basicConfig(filename="logs/"+args.net+"_"+args.dataset+"_"+str(args.session)+".log",
                        filemode='w', level=logging.DEBUG)
    logging.info(str(datetime.now()))

    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    if args.dataset == "SKA_detect":
        args.imdb_name = "traindata300-300"
        args.imdbval_name = "testdata300-300"
        # args.set_cfgs = ['FPN_ANCHOR_SCALES', '[1, 2, 4, 8, 16]', 'FPN_FEAT_STRIDES', '[1, 2, 4, 8, 16]',\
        #                  'MAX_NUM_GT_BOXES', '2500']
        args.set_cfgs = None

    args.cfg_file = "cfgs/{}.yml".format(args.net)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print('Using config:')
    pprint.pprint(cfg)
    logging.info(cfg)
    np.random.seed(cfg.RNG_SEED)

    # train set
    # -- Note: Use validation set and disable the flipped to enable faster loading.
    cfg.TRAIN.USE_FLIPPED = 1
    cfg.USE_GPU_NMS = args.cuda
    imdb_train, roidb_train, ratio_list_train, ratio_index_train = combined_roidb(args.imdb_name,cfg.TRAIN.USE_FLIPPED)
    train_size = len(roidb_train)
    _print('Train: {:d} roidb entries'.format(len(roidb_train)), logging)

    train_dataset = roibatchLoader(roidb_train, ratio_list_train, ratio_index_train, args.batch_size, \
                             imdb_train.num_classes, training=True)
    sampler_batch = sampler(train_size, args.batch_size)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                             sampler=sampler_batch, num_workers=args.num_workers)

    ### test set
    cfg.TRAIN.USE_FLIPPED = False
    imdb_test, roidb_test, ratio_list_test, ratio_index_test = combined_roidb(args.imdbval_name,cfg.TRAIN.USE_FLIPPED, False)
    print('Test:{:d} roidb entries'.format(len(roidb_test)))

    output_dir = args.save_dir + "/" + args.net + "/" + args.dataset
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # initilize the tensor holder here.
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)

    # ship to cuda
    if args.cuda:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()

    # make variable

    im_info = Variable(im_info)
    num_boxes = Variable(num_boxes)
    gt_boxes = Variable(gt_boxes)

    if args.cuda:
        cfg.CUDA = True

    # initilize the network here.
    if args.net == 'res101':
        FPN = resnet(imdb_train.classes, 101, pretrained=False, class_agnostic=args.class_agnostic)
    elif args.net == 'res50':
        FPN = resnet(imdb_train.classes, 50, pretrained=False, class_agnostic=args.class_agnostic,channel = args.channel)
    elif args.net == 'detnet59':
        FPN = detnet(imdb_train.classes, 59, pretrained=True, class_agnostic=args.class_agnostic)
    else:
        print("network is not defined")
        pdb.set_trace()
    print(FPN)
    FPN.create_architecture(args.channel)

    lr = args.lr
    momentum = cfg.TRAIN.MOMENTUM
    weight_decay = cfg.TRAIN.WEIGHT_DECAY

    params = []
    for key, value in dict(FPN.named_parameters()).items():
        if value.requires_grad:
            if 'bias' in key:
                params += [{'params':[value],'lr':lr*(cfg.TRAIN.DOUBLE_BIAS + 1), \
                            'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
            else:
                params += [{'params':[value],'lr':lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

    if args.cuda:
        FPN.cuda()

    if args.optimizer == "adam":
        lr = lr * 0.1
        optimizer = torch.optim.Adam(params)

    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)

    if args.resume:
        load_name = os.path.join(output_dir,
                                 'fpn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))
        _print("loading checkpoint %s" % (load_name), logging)
        checkpoint = torch.load(load_name)
        args.session = checkpoint['session']
        args.start_epoch = checkpoint['epoch']
        FPN.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr = optimizer.param_groups[0]['lr']
        weight_decay = optimizer.param_groups[0]['weight_decay']
        double_bias = True
        bias_decay = True
        params = []
        for key, value in dict(FPN.named_parameters()).items():
            if value.requires_grad:
                if 'bias' in key:
                    params += [{'params': [value], 'lr': lr * (double_bias + 1), \
                                'weight_decay': bias_decay and weight_decay or 0}]
                else:
                    params += [{'params': [value], 'lr': lr, 'weight_decay': weight_decay}]

        optimizer = torch.optim.Adam(params)
        _print("loaded checkpoint %s" % (load_name), logging)

    if args.mGPUs:
        if torch.cuda.device_count() > 1:
            device_ids = [0, 1]
            FPN = nn.DataParallel(FPN.cuda(), device_ids=device_ids).cuda()

    iters_per_epoch = int(train_size / args.batch_size)

    map = []
    best_map = 0.0

    if args.use_scheduler:
        scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,args.max_epochs)
        scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=8, total_epoch=int(args.max_epochs/4),
                                                  after_scheduler=scheduler_cosine)

    for epoch in range(args.start_epoch, args.max_epochs):
        # setting to train mode
        FPN.train()
        loss_temp = 0
        train_start = time.time()

        if args.use_scheduler:
            scheduler_warmup.step(epoch)   # 10 epoch warmup, after that schedule as scheduler_plateau
        else:
            if epoch % (args.lr_decay_step + 1) == 0:
                adjust_learning_rate(optimizer, args.lr_decay_gamma)
                lr *= args.lr_decay_gamma

        train_data_iter = iter(train_dataloader)

        for step in range(iters_per_epoch):
            data = train_data_iter.next()   #迭代器
            if (args.torch1_1):
                im_data.resize_(data[0].size()).copy_(data[0])
                im_info.resize_(data[1].size()).copy_(data[1])
                gt_boxes.resize_(data[2].size()).copy_(data[2])
                num_boxes.resize_(data[3].size()).copy_(data[3])
            else:
                im_data.data.resize_(data[0].size()).copy_(data[0])
                im_info.data.resize_(data[1].size()).copy_(data[1])
                gt_boxes.data.resize_(data[2].size()).copy_(data[2])
                num_boxes.data.resize_(data[3].size()).copy_(data[3])
            FPN.zero_grad()


            _, _, _, rpn_loss_cls, rpn_loss_box, \
            RCNN_loss_cls, RCNN_loss_bbox, \
            roi_labels = FPN(im_data, im_info, gt_boxes, num_boxes)

            if args.Use_regularization:
                weight_decay = 0.00005
                if weight_decay > 0:
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    reg_loss = Regularization(FPN, weight_decay).to(device)
                else:
                    print("no regularization")
                regloss = reg_loss(FPN)
                # print("regloss:",regloss)
                # RCNN_loss_cls = RCNN_loss_cls + regloss

            loss = rpn_loss_cls.mean() + rpn_loss_box.mean() \
                   + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean() + regloss
            loss_temp = loss_temp + loss.item()

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % args.disp_interval == 0:
                end = time.time()
                if step > 0:
                    loss_temp = loss_temp/args.disp_interval

                if args.mGPUs:
                    loss_rpn_cls = rpn_loss_cls.mean().data[0]
                    loss_rpn_box = rpn_loss_box.mean().data[0]
                    loss_rcnn_cls = RCNN_loss_cls.mean().data[0]
                    loss_rcnn_box = RCNN_loss_bbox.mean().data[0]
                    fg_cnt = torch.sum(roi_labels.data.ne(0))
                    bg_cnt = roi_labels.data.numel() - fg_cnt
                else:
                    loss_rpn_cls = rpn_loss_cls.item()
                    loss_rpn_box = rpn_loss_box.item()
                    loss_rcnn_cls = RCNN_loss_cls.item()
                    loss_rcnn_box = RCNN_loss_bbox.item()
                    fg_cnt = torch.sum(roi_labels.data.ne(0))
                    bg_cnt = roi_labels.data.numel() - fg_cnt

                if args.use_scheduler:
                    _print("[session %d][epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e" \
                           % (args.session, epoch, step,iters_per_epoch, loss_temp, optimizer.param_groups[0]['lr']), logging)
                else:
                    _print("[session %d][epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e" \
                        % (args.session, epoch, step,iters_per_epoch, loss_temp, lr), logging)
                _print("\t\t\tfg/bg=(%d/%d), time cost: %f" % (fg_cnt, bg_cnt, end - train_start), logging)
                _print("\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f,reg_loss %.4f, " \
                         % (loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box,regloss), logging)

                loss_temp = 0
                train_start = time.time()


        train_end = time.time()
        print("train time: %0.4fs" % (train_end - train_start))

        test_start = time.time()

        ###  begin to test
        thresh = 0.5
        max_per_image = 2500

        save_name_test  = 'faster_rcnn_10'
        # num_images = len(imdb_test.image_index)   # 自改
        num_images = len(roidb_test)
        all_boxes = [[[] for _ in range(num_images)]
                     for _ in range(imdb_test.num_classes)]  # 初始化

        output_dir_in_test = get_output_dir(imdb_test, save_name_test)
        test_dataset = roibatchLoader(roidb_test, ratio_list_test, ratio_index_test, args.test_batch_size, \
                                 imdb_test.num_classes, training=False, normalize=False)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size,
                                                 shuffle=False, num_workers=0,
                                                 pin_memory=True)

        test_data_iter = iter(test_dataloader)
        _t = {'im_detect': time.time(), 'misc': time.time()}
        det_file = os.path.join(output_dir_in_test, 'detections.pkl')

        FPN.eval()
        empty_array = np.transpose(np.array([[], [], [], [], []]), (1, 0))
        for i in range(num_images):
            data = test_data_iter.next()
            if (args.torch1_1):
                im_data.resize_(data[0].size()).copy_(data[0])
                im_info.resize_(data[1].size()).copy_(data[1])
                gt_boxes.resize_(data[2].size()).copy_(data[2])
                num_boxes.resize_(data[3].size()).copy_(data[3])
            else:
                im_data.data.resize_(data[0].size()).copy_(data[0])
                im_info.data.resize_(data[1].size()).copy_(data[1])
                gt_boxes.data.resize_(data[2].size()).copy_(data[2])
                num_boxes.data.resize_(data[3].size()).copy_(data[3])

            det_tic = time.time()

            rois, cls_prob, bbox_pred, \
            _, _, _, _, _ = FPN(im_data, im_info, gt_boxes, num_boxes)

            scores = cls_prob.data
            boxes = rois.data[:, :, 1:5]

            if cfg.TEST.BBOX_REG:
                # Apply bounding-box regression deltas
                box_deltas = bbox_pred.data
                if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                    # Optionally normalize targets by a precomputed mean and stdev
                    if args.class_agnostic:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                     + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                        box_deltas = box_deltas.view(1, -1, 4)
                    else:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                     + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                        box_deltas = box_deltas.view(1, -1, 4 * len(imdb_test.classes))

                pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
                pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
            else:
                # Simply repeat the boxes, once for each class
                pred_boxes = boxes

            pred_boxes /= data[1][0][2].cuda()

            scores = scores.squeeze()
            pred_boxes = pred_boxes.squeeze()
            det_toc = time.time()
            detect_time = det_toc - det_tic
            misc_tic = time.time()
            label = list()

            # 遍历除背景外的所有类别，取出每个类别score > thresh( = 0.5)的所有框，NMS(阈值默认为0.2)
            for j in range(1, imdb_test.num_classes):
                inds = torch.nonzero(scores[:, j] > thresh).view(-1)
                # if there is det
                if inds.numel() > 0:
                    cls_scores = scores[:, j][inds]
                    _, order = torch.sort(cls_scores, 0, True)
                    if args.class_agnostic:
                        cls_boxes = pred_boxes[inds, :]
                    else:
                        cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]

                    cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                    cls_dets = cls_dets[order]
                    keep = nms(cls_dets.cpu())  ##　use soft_nms
                    # keep = nms(cls_dets, cfg.TEST.NMS)
                    ###  error : data type not understood
                    # cls_dets = cls_dets([keep.view(-1).long()])
                    cls_dets = keep

                    if args.vis:
                        im2show = vis_detections(im2show, imdb_test.classes[j], cls_dets, 0.2)  # 可可视化！
                    all_boxes[j][i] = cls_dets
                else:
                    all_boxes[j][i] = empty_array

            # Limit to max_per_image detections *over all classes*
            if max_per_image > 0:
                image_scores = np.hstack([all_boxes[j][i][:, -1]
                                          for j in range(1, imdb_test.num_classes)])
                if len(image_scores) > max_per_image:
                    image_thresh = np.sort(image_scores)[-max_per_image]
                    for j in range(1, imdb_test.num_classes):
                        keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                        all_boxes[j][i] = all_boxes[j][i][keep, :]

            misc_toc = time.time()
            nms_time = misc_toc - misc_tic

            sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r' \
                             .format(i + 1, num_images, detect_time, nms_time))
            sys.stdout.flush()

        with open(det_file, 'wb') as f:
            cPickle.dump(all_boxes, f, cPickle.HIGHEST_PROTOCOL)


        print("number {} epoch's Evaluating detections".format(epoch))
        recs,precs,mean_ap = imdb_test.evaluate_detections(all_boxes, output_dir_in_test)
        _print("number  %2d epoch 's map: %0.4f" % (epoch , mean_ap), logging)
        # 保存比上次map值好的模型
        if mean_ap >= best_map:
            best_map = mean_ap
            save_name = os.path.join(output_dir, 'fpn_{}_{}_{}.pth'.format(args.session, epoch, step))
            save_checkpoint({
                'session': args.session,
                'epoch': epoch ,
                'model': FPN.module.state_dict() if args.mGPUs else FPN.state_dict(),
                'optimizer': optimizer.state_dict(),
                'pooling_mode': cfg.POOLING_MODE,
                'class_agnostic': args.class_agnostic,
            }, save_name)
            _print('save model: {}'.format(save_name), logging)
            ### draw P-R curve
            for key in recs.keys():
                plt.figure()
                plt.ylim(0, 1.1)
                plt.xlabel("Recall")
                plt.xlim(0, 1.1)
                plt.ylabel("Precison")
                plt.plot(recs[key], precs[key], label="P-R")
                plt.title("Prec-Rec")
                plt.legend(loc='upper right')
                plt.savefig('../label_de/output/visiual_map_loss/session{}_{}_PR.jpg'.format(args.session,key))
                plt.close()
        map.append(mean_ap)

        test_end = time.time()
        print("test time: %0.4fs" % (test_end - test_start))

    plt.figure()
    epoch_list = [x for x in range(0,len(map),1)]
    # plt.xticks(np.arange(0, len(map),1), epoch_list)
    plt.plot(epoch_list,map, color="red", label="test_map")
    plt.title("map-epoch")
    plt.legend(loc='lower right')
    plt.savefig('../label_de/output/visiual_map_loss/session{}test_map.jpg'.format(args.session))
    plt.close()
    print("completed!!!")













