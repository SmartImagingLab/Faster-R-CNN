import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, gradcheck
from torch.autograd.gradcheck import gradgradcheck
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
import torchvision.utils as vutils
from model.utils.config import cfg
from model.rpn.rpn_fpn import _RPN_FPN
# from model.roi_pooling.modules.roi_pool import _RoIPooling
# from model.roi_crop.modules.roi_crop import _RoICrop
# from model.roi_align.modules.roi_align import RoIAlignAvg
from model.AlignPool.roi_util.roi_align import ROIAlign as RoIAlignAvg
from model.AlignPool.roi_util.roi_pool import ROIPool as _RoIPooling
from model.rpn.proposal_target_layer import _ProposalTargetLayer
from model.utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, _affine_theta
from model.utils.net_utils import FocalLoss
from model.utils.label_smooth import LabelSmoothSoftmaxCE
from model.utils.generalized_iou_loss import generalized_iou_loss
from model.utils.Giou_loss import Giou_np
from model.rpn.bbox_transform import bbox_transform_inv
from model.rpn.bbox_transform import clip_boxes
import time
import pdb
from model.fpn.CBAM import ChannelAttention,SpatialAttention,CBAM


class _FPN(nn.Module):
    """ FPN """
    def __init__(self, classes,channel,class_agnostic):
        super(_FPN, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.class_agnostic = class_agnostic
        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0

        self.maxpool2d = nn.MaxPool2d(1, stride=2)
        # define rpn
        self.RCNN_rpn = _RPN_FPN(self.dout_base_model)
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)

        # NOTE: the original paper used pool_size = 7 for cls branch, and 14 for mask branch, to save the
        # computation time, we first use 14 as the pool_size, and then do stride=2 pooling for cls branch.
        self.RCNN_roi_pool = _RoIPooling((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0/16.0)   ## ori:1/16.0
        self.RCNN_roi_align = RoIAlignAvg((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0/16.0 , 2)   ## ori:1/16.0
        self.grid_size = cfg.POOLING_SIZE * 2 if cfg.CROP_RESIZE_WITH_MAX_POOL else cfg.POOLING_SIZE
        # self.RCNN_roi_crop = _RoICrop()
        self.row_sapmle = nn.Conv2d(channel, channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #self.up_sample_1 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.up_sample_2 = nn.Conv2d(channel, channel, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.up_sample_3 = nn.Conv2d(channel, channel, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.up_sample_4 = nn.Conv2d(channel, channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.smooth2 = nn.Conv2d(channel, channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.smooth3 = nn.Conv2d(channel, channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.smooth4 = nn.Conv2d(channel, channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        # custom weights initialization called on netG and netD
        def weights_init(m, mean, stddev, truncated=False):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.fill_(0)
            elif classname.find('BatchNorm') != -1:
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

        normal_init(self.RCNN_toplayer, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_smooth1, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_smooth2, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_smooth3, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_latlayer1, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_latlayer2, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_latlayer3, 0, 0.01, cfg.TRAIN.TRUNCATED)

        normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)
        weights_init(self.RCNN_top, 0, 0.01, cfg.TRAIN.TRUNCATED)

        normal_init(self.row_sapmle,0,0.01,cfg.TRAIN.TRUNCATED)
        normal_init(self.up_sample_2,0,0.01,cfg.TRAIN.TRUNCATED)
        normal_init(self.up_sample_3,0,0.01,cfg.TRAIN.TRUNCATED)
        normal_init(self.up_sample_4,0,0.01,cfg.TRAIN.TRUNCATED)
        normal_init(self.up_sample_4,0,0.01,cfg.TRAIN.TRUNCATED)
        normal_init(self.smooth2,0,0.01,cfg.TRAIN.TRUNCATED)
        normal_init(self.smooth3,0,0.01,cfg.TRAIN.TRUNCATED)
        normal_init(self.smooth4,0,0.01,cfg.TRAIN.TRUNCATED)

    def create_architecture(self,channel):
        self._init_modules(channel)
        self._init_weights()

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        #return torch.cat([x,y],dim=0)
        _,_,H,W = y.size()
        return F.upsample(x, size=(H,W), mode='bilinear') + y


    def downsample_add(self,x,y):
        # print(x.shape)
        # print(y.shape)
        #print(torch.cat([x,y],dim=0).shape)
        #downsample = torch.cat([x,y],dim=0)
        #downsample = torch.reshape(downsample,dim = 0)
        #print(downsample.shape)
        #print((x+y).shape)
        return x+y

    def _PyramidRoI_Feat(self, feat_maps, rois, im_info):
        ''' roi pool on pyramid feature maps'''
        # do roi pooling based on predicted rois
        img_area = im_info[0][0] * im_info[0][1]
        h = rois.data[:, 4] - rois.data[:, 2] + 1
        w = rois.data[:, 3] - rois.data[:, 1] + 1

        ### 不同尺度的ROI使用不同特征层作为ROI pooling层的输入
        # roi_level = torch.log(torch.sqrt(h * w) / 224.0)
        # roi_level = torch.round(roi_level + 4)
        # roi_level[roi_level < 2] = 2
        # roi_level[roi_level > 5] = 5

        # ###  直接选择P2特征层   2019.4.24.20:17
        roi_level = torch.log(torch.sqrt(h * w) / 512)     # can alter
        roi_level = torch.round(roi_level + 4)
        roi_level[roi_level < 2] = 2
        roi_level[roi_level > 5] = 5


        # roi_level.fill_(5)
        if cfg.POOLING_MODE == 'crop':
            # pdb.set_trace()
            # pooled_feat_anchor = _crop_pool_layer(base_feat, rois.view(-1, 5))
            # NOTE: need to add pyrmaid
            grid_xy = _affine_grid_gen(rois,  feat_maps.size()[2:], self.grid_size)
            grid_yx = torch.stack([grid_xy.data[:,:,:,1], grid_xy.data[:,:,:,0]], 3).contiguous()
            roi_pool_feat = self.RCNN_roi_crop(feat_maps, Variable(grid_yx).detach())
            if cfg.CROP_RESIZE_WITH_MAX_POOL:
                roi_pool_feat = F.max_pool2d(roi_pool_feat, 2, 2)

        elif cfg.POOLING_MODE == 'align':
            roi_pool_feats = []
            box_to_levels = []
            for i, l in enumerate(range(2, 6)):
                if (roi_level == l).sum() == 0:
                    continue
                # idx_l = (roi_level == l).nonzero().squeeze()
                idx_l = (roi_level == l).nonzero()
                if idx_l.shape[0] > 1:
                    idx_l = idx_l.squeeze()
                else:
                    idx_l = idx_l.view(-1)
                box_to_levels.append(idx_l)
                scale = feat_maps[i].size(2) / im_info[0][0]
                feat = self.RCNN_roi_align(feat_maps[i], rois[idx_l],scale)
                roi_pool_feats.append(feat)
            roi_pool_feat = torch.cat(roi_pool_feats, 0)
            box_to_level = torch.cat(box_to_levels, 0)
            idx_sorted, order = torch.sort(box_to_level)
            roi_pool_feat = roi_pool_feat[order]

        elif cfg.POOLING_MODE == 'pool':
            roi_pool_feats = []
            box_to_levels = []
            for i, l in enumerate(range(2, 6)):
                if (roi_level == l).sum() == 0:
                    continue
                idx_l = (roi_level == l).nonzero().squeeze()
                box_to_levels.append(idx_l)
                scale = feat_maps[i].size(2) / im_info[0][0]
                feat = self.RCNN_roi_pool(feat_maps[i], rois[idx_l],scale)
                roi_pool_feats.append(feat)
            roi_pool_feat = torch.cat(roi_pool_feats, 0)
            box_to_level = torch.cat(box_to_levels, 0)
            idx_sorted, order = torch.sort(box_to_level)
            roi_pool_feat = roi_pool_feat[order]
            
        return roi_pool_feat

    # def up_add(x,y):
    #     _, _, H, W = y.size()
    #     return F.upsample(x, size=(H, W), mode='bilinear') + y

    def _CBAM_(self,data):
        cbam_net = CBAM(data.shape[1])
        cbam_net = cbam_net.cuda()
        out = cbam_net(data)
        return out

    def forward(self, im_data, im_info, gt_boxes, num_boxes):
        batch_size = im_data.size(0)

        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data

        # feed image data to base model to obtain base feature map
        # Bottom-up
        c1 = self.RCNN_layer0(im_data)
        c1 = self._CBAM_(c1)
        c2 = self.RCNN_layer1(c1)
        c2 = self._CBAM_(c2)
        c3 = self.RCNN_layer2(c2)
        c3 = self._CBAM_(c3)
        c4 = self.RCNN_layer3(c3)
        c4 = self._CBAM_(c4)
        c5 = self.RCNN_layer4(c4)
        c5 = self._CBAM_(c5)
        # Top-down
        p5 = self.RCNN_toplayer(c5)
        p4 = self._upsample_add(p5, self.RCNN_latlayer1(c4))
        p4 = self.RCNN_smooth1(p4)
        p3 = self._upsample_add(p4, self.RCNN_latlayer2(c3))
        p3 = self.RCNN_smooth2(p3)
        p2 = self._upsample_add(p3, self.RCNN_latlayer3(c2))
        p2 = self.RCNN_smooth3(p2)

        p6 = self.maxpool2d(p5)

        ########################################################
        #row_sapmle = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #up_sample_1 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        # up_sample_2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        # up_sample_3 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        # up_sample_4 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        #smooth1 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        PAN2 = p2


        PAN3 = self.downsample_add(self.up_sample_2(PAN2),self.row_sapmle(p3))
        PAN3 = self.smooth2(PAN3)

        PAN4 = self.downsample_add(self.up_sample_3(PAN3),self.row_sapmle(p4))
        PAN4 = self.smooth3(PAN4)

        PAN5 = self.downsample_add(self.up_sample_4(PAN4),self.row_sapmle(p5))
        PAN5 = self.smooth4(PAN5)

        PAN6 = self.maxpool2d(PAN5)


        # rpn_feature_maps = [p2, p3, p4, p5, p6]
        # mrcnn_feature_maps = [p2, p3, p4, p5]
        rpn_feature_maps = [PAN2,PAN3,PAN4,PAN5,PAN6]
        mrcnn_feature_maps = [PAN2,PAN3,PAN4,PAN5]

        #########################################
        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(rpn_feature_maps, im_info, gt_boxes, num_boxes)

        # if it is training phrase, then use ground trubut bboxes for refining
        if self.training:
            roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
            rois, rois_label, gt_assign, rois_target, rois_inside_ws, rois_outside_ws = roi_data

            ## NOTE: additionally, normalize proposals to range [0, 1],
            #        this is necessary so that the following roi pooling
            #        is correct on different feature maps
            # rois[:, :, 1::2] /= im_info[0][1]
            # rois[:, :, 2::2] /= im_info[0][0]

            rois = rois.view(-1, 5)
            rois_label = rois_label.view(-1).long()
            gt_assign = gt_assign.view(-1).long()
            pos_id = rois_label.nonzero().squeeze()
            gt_assign_pos = gt_assign[pos_id]
            rois_label_pos = rois_label[pos_id]
            rois_label_pos_ids = pos_id

            rois_pos = Variable(rois[pos_id])
            rois = Variable(rois)
            rois_label = Variable(rois_label)

            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
        else:
            ## NOTE: additionally, normalize proposals to range [0, 1],
            #        this is necessary so that the following roi pooling
            #        is correct on different feature maps
            # rois[:, :, 1::2] /= im_info[0][1]
            # rois[:, :, 2::2] /= im_info[0][0]

            rois_label = None
            gt_assign = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0
            rois = rois.view(-1, 5)
            pos_id = torch.arange(0, rois.size(0)).long().type_as(rois).long()
            rois_label_pos_ids = pos_id
            rois_pos = Variable(rois[pos_id])
            rois = Variable(rois)

        # pooling features based on rois, output 14x14 map
        roi_pool_feat = self._PyramidRoI_Feat(mrcnn_feature_maps, rois, im_info)

        # feed pooled features to top model
        pooled_feat = self._head_to_tail(roi_pool_feat)


        # compute bbox offset
        bbox_pred = self.RCNN_bbox_pred(pooled_feat)
        if self.training and not self.class_agnostic:
            # select the corresponding columns according to roi labels
            bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
            bbox_pred_select = torch.gather(bbox_pred_view, 1, rois_label.long().view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
            bbox_pred = bbox_pred_select.squeeze(1)

        # compute object classification probability
        cls_score = self.RCNN_cls_score(pooled_feat)
        cls_prob = F.softmax(cls_score,dim=1)

        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0

        if self.training:
            # loss (cross entropy) for object classification
            Use_focal_loss = False
            Use_label_smoothing = False
            Use_Giou_loss = False
            if not Use_focal_loss:
                if Use_label_smoothing:
                    RCNN_loss_cls = LabelSmoothSoftmaxCE(cls_score, rois_label)
                else:
                    RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)
            else:
                FL = FocalLoss(class_num=self.n_classes, alpha = 1 ,gamma=2)
                RCNN_loss_cls = FL(cls_score,rois_label)
                RCNN_loss_cls = RCNN_loss_cls.type(torch.FloatTensor).cuda()

            # loss (l1-norm) for bounding box regression
            if Use_Giou_loss:
                rois1 = rois.view(batch_size, -1, rois.size(1))
                boxes = rois1.data[:,:, 1:5]
                bbox_pred1 = bbox_pred.view(batch_size, -1, bbox_pred.size(1))
                box_deltas = bbox_pred1.data
                pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
                pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
                pred_boxes /= im_info[0][2].cuda()
                # RCNN_loss_bbox = generalized_iou_loss(rois_target,bbox_pred)
                _,_,RCNN_loss_bbox = Giou_np(pred_boxes,boxes)
            else:
                RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)

        rois = rois.view(batch_size, -1, rois.size(1))
        cls_prob = cls_prob.view(batch_size, -1, cls_prob.size(1))
        bbox_pred = bbox_pred.view(batch_size, -1, bbox_pred.size(1))

        if self.training:
            rois_label = rois_label.view(batch_size, -1)

        return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label



