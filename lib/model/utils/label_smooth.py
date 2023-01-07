#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
import torch.nn as nn

#
# class LabelSmoothSoftmaxCE(nn.Module):
#     def __init__(self,
#                  lb_pos=0.9,
#                  lb_neg=0.005,
#                  reduction='mean',
#                  lb_ignore=255,
#                  ):
#         super(LabelSmoothSoftmaxCE, self).__init__()
#         self.lb_pos = lb_pos
#         self.lb_neg = lb_neg
#         self.reduction = reduction
#         self.lb_ignore = lb_ignore
#         self.log_softmax = nn.LogSoftmax(1)
#
#     def forward(self, logits, label):
#
#         logs = self.log_softmax(logits)
#         # ignore = label.data.cpu() == self.lb_ignore
#         ignore = label.data.cpu()
#         n_valid = (ignore == 0).sum()
#         label[ignore] = 0
#         lb_one_hot = logits.data.clone().zero_().scatter_(1, label.unsqueeze(1), 1)
#         label = self.lb_pos * lb_one_hot + self.lb_neg * (1-lb_one_hot)
#         ignore = ignore.nonzero()
#         _, M = ignore.size()
#         a, *b = ignore.chunk(M, dim=1)   ## 数组分块
#         label[[a, torch.arange(label.size(1)).type(torch.uint8), *b]] = 0
#         n_valid = n_valid.float().cuda()
#
#         if self.reduction == 'mean':
#             loss = -torch.sum(torch.sum(logs*label, dim=1)) / n_valid
#         elif self.reduction == 'none':
#             loss = -torch.sum(logs*label, dim=1)
#         return loss

def LabelSmoothSoftmaxCE(logits,label, lb_pos=0.9,lb_neg=0.005,reduction='mean'):
    log_softmax = nn.LogSoftmax(dim=1)
    logs = log_softmax(logits)
    # ignore = label.data.cpu() == self.lb_ignore
    ignore = label.data.cpu()
    n_valid = (ignore == 0).sum()
    label[ignore] = 0
    lb_one_hot = logits.data.clone().zero_().scatter_(1, label.unsqueeze(1), 1)
    label = lb_pos * lb_one_hot + lb_neg * (1 - lb_one_hot)
    ignore = ignore.nonzero()
    _, M = ignore.size()
    a, *b = ignore.chunk(M, dim=1)  ## 数组分块
    # label[[a, torch.arange(label.size(1)).type(torch.uint8), *b]] = 0
    n_valid = n_valid.float().cuda()
    if reduction == 'mean':
        loss = -torch.sum(torch.sum(logs * label, dim=1)) / n_valid
    elif reduction == 'none':
        loss = -torch.sum(logs * label, dim=1)

    return loss


if __name__ == '__main__':
    torch.manual_seed(15)
    out = torch.randn(10, 5).cuda()
    lbs = torch.randint(5, (10,)).cuda().long()
    print('out:', out)
    print('lbs:', lbs)

    import torch.nn.functional as F

    loss = LabelSmoothSoftmaxCE(out, lbs)
    print('loss:', loss)



