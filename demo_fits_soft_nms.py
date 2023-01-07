# ------------------------------------------------------------------------------------------
# The pytorch demo code for detecting the object in a specific image (fpn specific version)
# Licensed under The MIT License [see LICENSE for details]
# Written by Jianwei Yang, modified by Wenbo Liu, based on code from faster R-CNN
# ------------------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pandas as pd
import _init_paths
from model.utils.blob import im_list_to_blob
import os
import sys
import numpy as np

np.set_printoptions(suppress=True)
import argparse
import pprint
import pdb
import pandas as pd
import time
import cv2
import matplotlib.pyplot as plt
import pickle as cPickle
import torch
from numpy.core.multiarray import ndarray
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
# from scipy.misc import imread
from imageio import imread
from roi_data_layer.roidb_v1 import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
from model.soft_nms.nms import soft_nms
from model.soft_nms.nms import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import vis_detections
# from model.fpn.fpn_cascade import _FPN
from model.fpn.resnet_IN import resnet
from astropy.io import fits
import pdb
import warnings
import csv
warnings.filterwarnings('ignore')

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--webcam_num', dest='webcam_num',
                        help='webcam ID number',
                        default=-1, type=int)
    parser.add_argument('--dataset', dest='dataset',
                        help='training dataset',
                        default='SKA_detect', type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfgs/res50.yml', type=str)
    parser.add_argument('--net', dest='net',
                        help='vgg16, res50, res101, res152',
                        default='res50', type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--load_dir', dest='load_dir',
                        help='directory to load models', default="../model_save",  # 可改
                        nargs=argparse.REMAINDER)
    parser.add_argument('--cuda', dest='cuda',
                        help='whether use CUDA', default=1,
                        action='store_true')
    parser.add_argument('--image_dir', dest='image_dir',
                        help='directory to load images for demo',
                        #default="/home/lab30202/sdd/lwb2/label_de/validation")  # 可改/home/lab30202/sdd/lwb2/label_de
                        default="/home/dell460/slc/sdd_01/Faster_mul/validation")  # 可改/home/lab30202/sdd/lwb2/label_de
    parser.add_argument('--cag', dest='class_agnostic',
                        help='whether perform class_agnostic bbox regression',
                        action='store_true')  # 类不可知？
    parser.add_argument('--parallel_type', dest='parallel_type',
                        help='which part of model to parallel, 0: all, 1: model before roi pooling',
                        default=0, type=int)
    parser.add_argument('--checksession', dest='checksession',
                        help='checksession to load model',
                        default=22, type=int)
    parser.add_argument('--checkepoch', dest='checkepoch',
                        help='checkepoch to load network',
                        default=255, type=int)
    parser.add_argument('--checkpoint', dest='checkpoint',
                        help='checkpoint to load network',
                        default=39, type=int)  # 可改
    parser.add_argument('--bs', dest='batch_size',
                        help='batch_size',
                        default=1, type=int)
    parser.add_argument('--lr', dest='lr',
                        help='starting learning rate',
                        default=0.001, type=float)
    parser.add_argument('--vis', dest='vis',
                        help='visualization mode',
                        default=True,
                        action='store_true')
    args = parser.parse_args()
    return args


def _get_image_blob(im):
    """Converts an image into a network input.
    Arguments:
      im (ndarray): a color image in BGR order
    Returns:
      blob (ndarray): a data blob holding an image pyramid
      im_scale_factors (list): list of image scales (relative to im) used
        in the image pyramid
    """
    im_orig = im.astype(np.float32, copy=True)
    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors)

# calculate pression and recall
def cal_indicators(save_path, det, flux_internal):
    csv_file = '/home/lab30202/sdd/lwb2/eventfile/' + save_path.split("/")[-1].split("_")[0] + '/obsinfo.csv'
    cmos = int(save_path.split("/")[-1].split(".")[0].split("_")[-1])
    df_label = pd.read_csv(csv_file)
    df_label_1 = df_label[(df_label['cmosnum'] == cmos)]  # .index.tolist()  # coms需要转成int
    df_label_1_x = df_label_1['x'].tolist()
    df_label_1_y = df_label_1['y'].tolist()
    df_label_1_flux = df_label_1['flux/mCrab'].tolist()
    n_real = len(df_label_1_y)
    n_se_num = len(flux_internal)
    recall_internal_num = np.zeros(n_se_num)
    sensitivity_internal_num = np.zeros(n_se_num)
    true_num = 0
    flag = np.zeros(n_real)
    file = open(save_path, 'w+')
    for t in range(len(det)):
        xm1, xm2 = (det[t, 0] * 16 + 1), (det[t, 2] * 16 + 1)
        ym1, ym2 = (det[t, 1] * 16 + 1), (det[t, 3] * 16 + 1)
        for i in range(n_real):
            if not flag[i]:  # 防止重复保存，一个目标至多被检测到一次（不会出现多个框都检测到目标而计数的情况）
                y0, x0 = df_label_1_y[i], df_label_1_x[i]  # 优先算作亮源！
                if xm1 <= x0 <= xm2 and ym1 <= y0 <= ym2:
                    flag[i] = 1
                    # 另一种保存方法
                    file.write("location {}(x,y):{:.4f} {:.4f}\n".format(i + 1, (xm1 + xm2) / 2, (ym1 + ym2) / 2))
                    true_num += 1
                    if df_label_1_flux[i] >= flux_internal[-1]:
                        recall_internal_num[-1] += 1
                    else:
                        for j in range(n_se_num - 1):
                            if df_label_1_flux[i] >= flux_internal[j] and df_label_1_flux[i] < flux_internal[j + 1]:
                                recall_internal_num[j] += 1
                    break  # 及时终止！一个框至多只算检测到一个目标，位置相近的目标可以用SOFT-NMS捕捉另外的框，或者调整阈值！

    # 计算查全\查准(合并重复的)\灵敏度
    if not true_num:
        file.write("no detected object！\n")
        print("There is no detected object in the %d image！\n", cmos)
    for j in range(n_se_num - 1):
        sensitivity_internal_num[j] = len(list(
            filter(lambda t: t >= flux_internal[j] and t < flux_internal[j + 1],
                   df_label_1_flux)))  # 滤除小于sensitivity的数目！
    sensitivity_internal_num[-1] = len(
        list(filter(lambda t: t >= flux_internal[-1], df_label_1_flux)))  # 滤除小于sensitivity的数目！
    precision_num = len(det)
    file.close()

    return true_num, recall_internal_num, precision_num, sensitivity_internal_num

def Cal_distance(point_a,point_b):
    p1 = point_a
    p2 = point_b
    squared_dist = np.sum((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 ,axis=0)
    dist = np.sqrt(squared_dist)
    return dist

def cal_indicators_3(im_file,det,labelpath):
    false_det = 0
    right_det = 0
    precision_num = 0
    label_num = 0
    # X = os.path.basename(im_file).split('_')[1]
    # Y = os.path.basename(im_file).split('_')[2].split('.')[0]
    det = det.reshape((-1,5))
    #global false_det,right_det,precision_all
    precision_num = det.shape[0]
    print(precision_num)
    #if not os.path.exists(labelpath +'/' +  'image_{}_{}.list'.format(X,Y)):
    if not os.path.exists(labelpath +'/' +  os.path.basename(im_file).split('.fits')[0] + 'list'):
        false_det = det.shape[0]
    else:
        #label = np.loadtxt(labelpath +'/' +  'image_{}_{}.list'.format(X,Y))
        label = np.loadtxt(labelpath +'/' + os.path.basename(im_file).split('.fits')[0] + 'list')
        label = label.reshape((-1,3))
        label_num = label.shape[0]
        print(label.shape[0])
        for i in range(det.shape[0]):
            center_x = (det[i][2] - det[i][0])/2 + det[i][0]
            center_y = (det[i][3] - det[i][1])/2 + det[i][1]
            flag_true = 0
            for j in range(label.shape[0]):
                if Cal_distance((center_x,center_y),(label[j][1],label[j][2]))<10:
                    right_det += 1
                    flag_true = 1
                    break
            if flag_true ==0:
                false_det += 1
    return precision_num,false_det,right_det,label_num

def Cal(path,labelpath):
    false_det_all = 0
    precision_all = 0
    right_det_all = 0
    label_all = 0
    filelist = glob.glob(os.path.join(path, "*.txt"))
    for file in filelist:
        pre_label = np.loadtxt(file)
        if not -1 in pre_label:
            precision, false_det, right_det,label_num = cal_indicators_2(file, pre_label, labelpath)
            precision_all += precision
            false_det_all += false_det
            print(right_det)
            right_det_all += right_det
            label_all += label_num
            #print(label_all)
    #print(precision_all,false_det_all,right_det_all)
    return precision_all,false_det_all,right_det_all,label_all

def cal_indicators_2(file,det,labelpath):
    false_det = 0
    right_det = 0
    label_num = 0
    # X = os.path.basename(file).split('_')[1]
    # Y = os.path.basename(file).split('_')[2]
    det = det.reshape((-1,5))
    precision_num = det.shape[0]
    #print(precision_num)
    if not os.path.exists(labelpath +'/' +  os.path.basename(file).split('_det.txt')[0] + '.list'):
        false_det = det.shape[0]
        print('错误检测',file)
        print('错误检测的数量',false_det)
    else:
        label = np.loadtxt(labelpath +'/' +  os.path.basename(file).split('_det.txt')[0] + '.list')
        label = label.reshape((-1,4))
        label_num = label.shape[0]
        #print(label.shape[0])
        for i in range(det.shape[0]):
            center_x = (det[i][2] - det[i][0])/2 + det[i][0]
            center_y = (det[i][3] - det[i][1])/2 + det[i][1]
            flag_true = 0
            for j in range(label.shape[0]):
                if Cal_distance((center_x,center_y),(label[j][1],label[j][2]))<10:
                    target_all_I.append(label[j][3])
                    right_det += 1
                    flag_true = 1
                    break
            if flag_true ==0:
                false_det += 1
                print('真实的但是没检测到',file)
    return precision_num,false_det,right_det,label_num

def Write_csv(thresh,precision_all,right_det_all,label_all):
    with open('/home/dell460/slc/sdd_01/Faster_mul/label_de/output/pre_recall.csv','a',encoding='utf-8') as file:
        f = csv.writer(file)
        nowtime = time.strftime('%Y-%m-%d %H:%M:%S %p',time.localtime())
        model = '{}_{}_{}'.format(args.checksession, args.checkepoch, args.checkpoint)
        pre = (right_det_all/precision_all)*100
        recall = (right_det_all/label_all)*100
        f.writerow((nowtime,
                    model,
                    thresh,
                    precision_all,
                    right_det_all,
                    '{:.2f}%'.format(pre),
                    '{:.2f}%'.format(recall)))
def Snr_plot(label_I,target_all_I,thresh):
    nums,bins = np.histogram(np.log2(label_I[:,2]),bins = 20)
    target_nums,target_bins = np.histogram(np.log2(np.unique(target_all_I)),bins = bins)
    pre_plot = []
    #df = pd.DataFrame({'bins':bins,'{}_recall'.format(thresh):recall})
    print(nums)
    print(target_nums)
    #print(np.arange(np.min(bins),np.max(bins),(np.max(bins)-np.min(bins))/20))
    for i in range(len(target_nums)):
        if target_nums[i] == nums[i]:
            pre_plot.append(1)
        else:
            pre_plot.append(target_nums[i]/nums[i])
    print(len(bins))
    print(len(pre_plot))
    df = pd.DataFrame({'bins':np.arange(np.min(bins),np.max(bins),(np.max(bins)-np.min(bins))/20),'{}_recall'.format(thresh):pre_plot})
    df.to_csv('./output/add_pan_cbam_mul_snr{}.csv'.format(thresh),index=False, sep=',')
    print('')
    # with open(r'/home/lab30201/sdd/slc/SKAData/Faster_mulchannel/label_de/output/add_pan_taylor_thresh_snr_4_.txt',mode='a',encoding='utf-8') as f:
    #     f.write(str(thresh).replace('[','').replace(']',''))
    #     f.write('\n')
    #     f.write(str(pre_plot))
    plt.plot((np.arange(np.min(bins),np.max(bins),(np.max(bins)-np.min(bins))/20)),pre_plot)
    plt.title('thresh={}'.format(thresh))
    plt.ylabel('recall')
    plt.xlabel('log_I(Jy)')
    plt.savefig('/home/dell460/slc/sdd_01/Faster_mul/label_de/output/22_addpan_addcbam_576_t1-t7_thresh={}_snr.png'.format(thresh))


if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print("GPU {} will be used\n".format("0"))

    args = parse_args()

    lr = args.lr
    momentum = cfg.TRAIN.MOMENTUM
    weight_decay = cfg.TRAIN.WEIGHT_DECAY

    print('Called with args:')
    print(args)
    cfg.USE_GPU_NMS = args.cuda
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print('Using config:')
    pprint.pprint(cfg)
    # 设置随机数种子
    # 每次运行代码时设置相同的seed，则每次生成的随机数也相同，
    # 如果不设置seed，则每次生成的随机数都会不一样
    np.random.seed(cfg.RNG_SEED)

    # load_dir 模型目录   args.net 网络   args.dataset 数据集
    input_dir = args.load_dir + "/" + args.net + "/" + args.dataset
    if not os.path.exists(input_dir):
        raise Exception('There is no input directory for loading network from ' + input_dir)
    load_name = os.path.join(input_dir,
                             'fpn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))
    print(load_name)

    pascal_classes = np.asarray(['__background__', 'SKA'])
    # initilize the network here.
    # class-agnostic 方式只回归2类bounding box，即前景和背景
    if args.net == 'res101':
        fpn = resnet(pascal_classes, 101, pretrained=False, class_agnostic=args.class_agnostic)
    elif args.net == 'res50':
        fpn = resnet(pascal_classes, 50, pretrained=False, class_agnostic=args.class_agnostic)
    elif args.net == 'res152':
        fpn = resnet(pascal_classes, 152, pretrained=False, class_agnostic=args.class_agnostic)
    else:
        print("network is not defined")
        # 到了pdb.set_trace()那就会定下来，就可以看到调试的提示符(Pdb)了
        pdb.set_trace()

    fpn.create_architecture()
    fpn.cuda()
    print("load checkpoint %s" % (load_name))
    if args.cuda > 0:
        checkpoint = torch.load(load_name)
    fpn.load_state_dict(checkpoint['model'])
    if 'pooling_mode' in checkpoint.keys():
        cfg.POOLING_MODE = checkpoint['pooling_mode']

    print('load model successfully!')

    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)
    # ship to cuda
    if args.cuda > 0:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()

    # make variable
    im_data = Variable(im_data)
    im_info = Variable(im_info)
    num_boxes = Variable(num_boxes)
    gt_boxes = Variable(gt_boxes)

    if args.cuda > 0:
        cfg.CUDA = True

    if args.cuda > 0:
        fpn.cuda()

    # 对dropout和batch normalization的操作在训练和测试的时候是不一样的
    # pytorch会自动把BN和DropOut固定住，不会取平均，而是用训练好的值
    fpn.eval()

    start = time.time()
    # max_per_image = 2500
    #thresh = 0.95
    thresh = 0.5
    vis = True
    webcam_num = args.webcam_num
    # Set up webcam or get image directories

    if webcam_num >= 0:
        cap = cv2.VideoCapture(webcam_num)  # 应该就是判断要不要自己用电脑录视频
        num_images = 0
    else:  # 如果不用电脑录视频，那么就读取image路径下的图片
        # imglist = os.listdir(args.image_dir)
        import glob
        fits_list = os.path.join(args.image_dir, '*.fits')
        imglist = glob.glob(fits_list)

        num_images = len(imglist)
    print('Loaded Photo: {} images.'.format(num_images))

    # cmos_all = []        # 记录图片的编号
    # flux_interval = np.linspace(0, 5, 11)
    # flux_interval = np.linspace(0, 10, 2)
    # all_recall = np.zeros(len(flux_interval))
    # all_flux_internal_num = np.zeros(len(flux_interval))
    # precision_all = 0
    # all_ture_num = 0
    false_det_all = 0
    precision_all = 0
    right_det_all = 0
    label_all = 0
    target_all_I = []
    labelpath = '/home/dell460/slc/sdd_01/Faster_mul/SKA0_I_random'
    while (num_images >= 0):
        box_list = list()
        total_tic = time.time()
        if webcam_num == -1:
            num_images -= 1

        # Get image from the webcam
        if webcam_num >= 0:
            if not cap.isOpened():
                raise RuntimeError("Webcam could not open. Please check connection.")
            # ret 为True 或者False,代表有没有读取到图片
            # frame表示截取到一帧的图片
            ret, frame = cap.read()
            im_in = np.array(frame)
        # Load the demo image
        else:
            im_file = imglist[num_images]
            # im_file = os.path.join(args.image_dir, imglist[num_images])

            img_fits = fits.open(im_file, ignore_missing_end=True)[0].data
            # ### use log transpoze
            # im = np.log(1 + np.abs(im))
            max_value = np.max(img_fits)
            min_value = np.min(img_fits)
            mean_value = np.mean(img_fits)
            var_value = np.var(img_fits)
            im_in = (img_fits - mean_value) / (max_value - min_value)
        # im_in = (im - mean_value)/var_value

        if len(im_in.shape) == 3:
            im_in = np.swapaxes(im_in, 0, 1)  # 8,256,256->256,8,256
            im = np.swapaxes(im_in, 1, 2)  # 256,8,256->256,256,8

        # if len(im_in.shape) == 2:
        #   im_in = im_in[:, :, np.newaxis]
        #   im_in = np.concatenate((im_in, im_in, im_in), axis=2)      # 扩展为三维数据
        # # rgb -> bgr
        # # line[:-1]其实就是去除了这行文本的最后一个字符（换行符）后剩下的部分。
        # # line[::-1]字符串反过来 line = "abcde" line[::-1] 结果为：'edcba'
        #im = im_in[:, :, ::-1]

        blobs, im_scales = _get_image_blob(im)  ##图片变换 该文件上面定义的函数，返回处理后的值 和尺度
        assert len(im_scales) == 1, "Only single-image batch implemented"
        im_blob = blobs
        im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)

        im_data_pt = torch.from_numpy(im_blob)
        im_data_pt = im_data_pt.permute(0, 3, 1, 2)  # 交换维度
        im_info_pt = torch.from_numpy(im_info_np)

        # 将tensor的大小调整为指定的大小。
        # 如果元素个数比当前的内存大小大，就将底层存储大小调整为与新元素数目一致的大小。
        im_data.resize_(im_data_pt.size()).copy_(im_data_pt)
        # print(im_data.shape)
        im_info.resize_(im_info_pt.size()).copy_(im_info_pt)
        # print(im_info.shape)
        gt_boxes.resize_(1, 5).zero_()  # 可以调整！
        # print(gt_boxes.shape)
        num_boxes.resize_(1).zero_()
        # print(num_boxes.shape)

        # pdb.set_trace()
        det_tic = time.time()

        rois, cls_prob, bbox_pred, \
        _, _, _, _, _ = fpn(im_data, im_info, gt_boxes, num_boxes)

        scores = cls_prob.data  # 读取值
        boxes = rois.data[:, :, 1:5]

        if cfg.TEST.BBOX_REG:  # 默认为 ture
            # Apply bounding-box regression deltas
            box_deltas = bbox_pred.data
            if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:  # 将预先计算好的目标归一化、规范化
                # Optionally normalize targets by a precomputed mean and stdev
                if args.class_agnostic:
                    if args.cuda > 0:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                     + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    else:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                                     + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)

                    box_deltas = box_deltas.view(1, -1, 4)
                else:
                    if args.cuda > 0:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                     + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    else:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                                     + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
                    box_deltas = box_deltas.view(1, -1, 4 * len(pascal_classes))

            # model.rpn.bbox_transform 根据anchor和偏移量计算proposals
            # 最后返回的是左上和右下顶点的坐标[x1,y1,x2,y2]。
            pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
            # 将改变坐标信息后超过图像边界的框的边框裁剪一下,使之在图像边界之内
            pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
        else:
            # Simply repeat the boxes, once for each class
            # #Numpy的 tile() 函数,就是将原矩阵横向、纵向地复制，这里是横向
            pred_boxes = np.tile(boxes, (1, scores.shape[1]))

        pred_boxes /= im_scales[0]

        scores = scores.squeeze()
        pred_boxes = pred_boxes.squeeze()
        det_toc = time.time()
        detect_time = det_toc - det_tic
        misc_tic = time.time()

        if vis:
            jpg_name = os.path.join(args.image_dir, im_file.split("/")[-1].split(".fits")[0] + ".png")
            im_jpg = cv2.imread(jpg_name)
            im_jpg = cv2.flip(im_jpg,0)
            im2show = np.copy(im_jpg)
            # new = np.empty((img_fits.shape[1],img_fits.shape[2]))
            # for z in range(img_fits.shape[0]):
            #     im2show += img_fits[z,:,:]
            # im2show = (np.log(img_fits[0, :, :] + 1))
            #im2show = (np.log(new + 1))

        for j in range(1, len(pascal_classes)):  # 对每个类别进行遍历画框
            inds = torch.nonzero(scores[:, j] > thresh).view(-1)  # 返回不为0的索引
            # if there is det
            if inds.numel() > 0:
                cls_scores = scores[:, j][inds]
                _, order = torch.sort(cls_scores, 0, True)
                if args.class_agnostic:
                    cls_boxes = pred_boxes[inds, :]
                else:
                    cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]

                cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
                cls_dets = cls_dets[order]
                keep = nms(cls_dets, 0.3)  # 0.3可行
                # keep = soft_nms(cls_dets)##　use soft_nms or nms  ——是否可以转化为cpu_nms()
                # keep = nms(cls_dets, cfg.TEST.NMS)
                ###  error : data type not understood
                # cls_dets = cls_dets([keep.view(-1).long()])
                cls_dets = keep

                if pascal_classes[j] == "SKA":  # 已改
                    class_name_index = 100
                    class_name_column = [class_name_index] * cls_dets.shape[0]
                    class_name = np.array(class_name_column).reshape(len(class_name_column), 1)
                    cls_dets = np.concatenate([cls_dets], axis=1)

                box_list.append(cls_dets)  # 一个类别：box_list==cls_dets
                if vis:
                    im2show, save_mat = vis_detections(im2show, pascal_classes[j], cls_dets,
                                                       thresh)  # 经过NMS之后的进一步置信度阈值设置-显示用！
        result_save_path = os.path.join(args.image_dir, imglist[num_images][:-4].rstrip(".") + ".save.txt")
        result_path = os.path.join(args.image_dir, imglist[num_images][:-4].rstrip(".") + "_det.txt")
        if not box_list:
            np.savetxt(result_path, np.array([-1]), fmt="%.8f")  # 若未检测到，则写入-1
        else:
            box_np = np.concatenate(box_list, axis=0)  # 按照列进行拼接！也可以是box_list
            save_mat = np.array(save_mat)  # 按照列进行拼接！也可以是box_list
            np.savetxt(result_path, box_np, fmt="%.8f")
            # 计算某一张图像中检测到的正确目标数量、flux每个区间中正确目标的数量、检测到的数量以及flux每个区间的本该找到的数量
            #precision,false_det,right_det,label = cal_indicators_3(im_file,box_np,labelpath)
            #                                                                                  labelpath)
            # all_ture_num += true_num  # 记录检测目标实际找到的总的正确目标的数量
            # all_recall += recall_internal_num  # 记录检测目标在flux每个区间中实际找到的正确目标的数量
            # all_flux_internal_num += flux_internal_num  # 记录flux每个区间的本该找到的数量
            # precision_all += precision # 记录总的检测到的数量
            # false_det_all += false_det
            # right_det_all += right_det
            # label_all += label
            # print(right_det_all)

        if vis and webcam_num == -1:
            result_path = os.path.join(args.image_dir, imglist[num_images][:-4] + "_det.png")
            cv2.imwrite(result_path, im2show)  # 写入图片内容到本地！
            #result_path_plt = os.path.join(args.image_dir, imglist[num_images][:-4] + "_detection.jpg")
            #plt.figure()
            #plt.imshow(np.log(im2show / np.max(im2show) + 9) * 255)
            #plt.savefig(result_path_plt)
        misc_toc = time.time()
        nms_time = misc_toc - misc_tic

    # 评价指标：
    # try:
    #     recall = all_recall / all_flux_internal_num  # 两个数组的除法！
    # except ZeroDivisionError:
    #     print("含有分母为0，改用依次遍历区间！")
    #     recall = np.ones(len(flux_interval))  # 初始化
    #     for j in range(len(flux_interval)):
    #         if not all_flux_internal_num[j]:
    #             recall[j] = all_recall[j] / all_flux_internal_num[j]
    # precision = all_ture_num / precision_all
    # print('session{}_{}_indicators: \n recall = {}\n precision = {:.2f}'.format(
    #     args.checksession, thresh, recall, precision))
    # 绘制曲线
    # plt.figure(figsize=(12, 12))
    # bar_width = 0.5
    # bar_width = 1
    # plt.bar(flux_interval, recall, bar_width, align='edge', color='r')  # align='center '
    # plt.xlabel('flux_interval')
    # plt.ylabel('recall')
    # plt.plot(flux_interval, recall, color="red", label="recall-img")
    # plt.legend(loc='best')
    # plt.title('indicators_img')
    # plt.savefig(
    #     './output/visiual_map_loss/session{}_{}_indicators_img.jpg'
    #         .format(args.checksession, thresh))
    # plt.close()
    ##precision_all,false_det_all,right_det_all = Cal(args.image_dir,labelpath)
    precision_all,false_det_all,right_det_all,label_all = Cal(args.image_dir,labelpath)
    label_I = np.loadtxt('/home/dell460/slc/sdd_01/Faster_mul/label_gleam_pixel_random_addI.txt')
    Snr_plot(label_I,target_all_I,thresh)
    print('thresh:',thresh)
    print('all_pre',precision_all)
    print('false_det_all',false_det_all)
    print('right_det_all',right_det_all)
    print('true_nums',label_all)
    print('查准率',right_det_all/precision_all)
    print('查全率',right_det_all/label_all)
    Write_csv(thresh,precision_all,right_det_all,label_all)
    print("completed!!!")