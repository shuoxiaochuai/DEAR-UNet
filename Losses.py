# -*- coding: utf-8 -*-
# @Time    : 2021/3/11 10:17 下午
# @Author  : Haonan Wang
# @File    : Losses.py
# @Software: PyCharm
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class WeightedBCE(nn.Module):

    def __init__(self, weights=[0.4, 0.6]):
        super(WeightedBCE, self).__init__()
        self.weights = weights

    def forward(self, logit_pixel, truth_pixel):
        logit = logit_pixel.view(-1)
        truth = truth_pixel.view(-1)
        assert(logit.shape==truth.shape)
        loss = F.binary_cross_entropy(logit, truth, reduction='none')
        pos = (truth>0.5).float()
        neg = (truth<0.5).float()
        pos_weight = pos.sum().item() + 1e-12
        neg_weight = neg.sum().item() + 1e-12
        loss = (self.weights[0]*pos*loss/pos_weight + self.weights[1]*neg*loss/neg_weight).sum()

        return loss

class WeightedDiceLoss(nn.Module):
    def __init__(self, weights=[0.2, 0.8]): # W_pos=0.8, W_neg=0.2
        super(WeightedDiceLoss, self).__init__()
        self.weights = weights

    def forward(self, logit, truth, smooth=1e-5):
        batch_size = len(logit)
        logit = logit.view(batch_size,-1)
        truth = truth.view(batch_size,-1)
        assert(logit.shape==truth.shape)
        p = logit.view(batch_size,-1)
        t = truth.view(batch_size,-1)
        w = truth.detach()
        w = w*(self.weights[1]-self.weights[0])+self.weights[0]
        # p = w*(p*2-1)  #convert to [0,1] --> [-1, 1]
        # t = w*(t*2-1)
        p = w*(p)
        t = w*(t)
        intersection = (p * t).sum(-1)
        union =  (p * p).sum(-1) + (t * t).sum(-1)
        dice  = 1 - (2*intersection + smooth) / (union +smooth)
        # print "------",dice.data

        loss = dice.mean()
        return loss

class WeightedDiceBCE(nn.Module):
    def __init__(self,dice_weight=1,BCE_weight=0.5):
        super(WeightedDiceBCE, self).__init__()
        self.BCE_loss = WeightedBCE(weights=[0.5, 0.5])
        self.dice_loss = WeightedDiceLoss(weights=[0.5, 0.5])
        self.BCE_weight = BCE_weight
        self.dice_weight = dice_weight
    def forward(self, inputs, targets):
        # inputs = inputs.contiguous().view(-1)
        # targets = targets.contiguous().view(-1)
        # print "dice_loss", self.dice_loss(inputs, targets)
        # print "focal_loss", self.focal_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        BCE = self.BCE_loss(inputs, targets)
        # print "dice",dice
        # print "focal",focal

        dice_BCE_loss = self.dice_weight * dice + self.BCE_weight * BCE
        # print "dice_focal",dice_focal_loss
        return dice_BCE_loss

class LBTW_Loss(nn.Module):

    def __init__(self, Loss):
        super(LBTW_Loss, self).__init__()
        self.criteria = Loss
        try:
            self.__name__ = "LBTW_" + self.criteria.__name__
        except:
            self.__name__ = "LBTW_" + self.criteria._get_name()
    def _get_name(self):
        super(LBTW_Loss, self)._get_name()
        return "LBTW_" + self.criteria._get_name()


    def forward(self, out, comb3, comb2, comb1, target):
        outs=[out,comb3,comb2,comb1]
        sub_loss = []
        for i in range(4):
            pred=outs[i]
            curr_loss = self.criteria(pred, target)
            sub_loss.append(curr_loss)
        # print sub_loss[0].data, sub_loss[1].data, sub_loss[2].data, sub_loss[3].data
        return sub_loss, sub_loss[0], sub_loss[1], sub_loss[2], sub_loss[3]

class LBTW_algorithm():
    def __init__(self):
        self.initial_taskout_loss_list = 0
        self.initial_task3_loss_list = 0
        self.initial_task2_loss_list = 0
        self.initial_task1_loss_list = 0
        self.weights_out_save = []
        self.weights_c3_save = []
        self.weights_c2_save = []
        self.weights_c1_save = []

    def __call__(self, batch_num,out_loss, c3_loss, c2_loss, c1_loss, alpha=0.5):
        if batch_num == 1:
            self.initial_taskout_loss_list = out_loss.item()
            self.initial_task3_loss_list = c3_loss.item()
            self.initial_task2_loss_list = c2_loss.item()
            self.initial_task1_loss_list = c1_loss.item()
        out_loss_ratio = out_loss.item() / self.initial_taskout_loss_list
        c3_loss_ratio = c3_loss.item() / self.initial_task3_loss_list
        c2_loss_ratio = c2_loss.item() / self.initial_task2_loss_list
        c1_loss_ratio = c1_loss.item() / self.initial_task1_loss_list

        out_loss_weight = pow(out_loss_ratio,alpha)
        c3_loss_weight = pow(c3_loss_ratio,alpha)
        c2_loss_weight = pow(c2_loss_ratio,alpha)
        c1_loss_weight = pow(c1_loss_ratio,alpha)

        weights_sum = out_loss_weight+c3_loss_weight+c2_loss_weight+c1_loss_weight
        out_loss_weight  = out_loss_weight / weights_sum * 4
        c3_loss_weight = c3_loss_weight / (weights_sum) * 4
        c2_loss_weight = c2_loss_weight / weights_sum * 4
        c1_loss_weight = c1_loss_weight / weights_sum * 4

        self.weights_out_save.append(out_loss_weight)
        self.weights_c3_save.append(c3_loss_weight)
        self.weights_c2_save.append(c2_loss_weight)
        self.weights_c1_save.append(c1_loss_weight)

        w0 = self.weights_out_save
        w3 = self.weights_c3_save
        w2 = self.weights_c2_save
        w1 = self.weights_c1_save
        losses = out_loss * out_loss_weight \
                 + c3_loss * c3_loss_weight \
                 + c2_loss * c2_loss_weight \
                 + c1_loss * c1_loss_weight
        return losses / 4.0, w0, w3, w2, w1


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        Focal Loss 用于解决难易样本不平衡
        :param alpha: 平衡系数，0<alpha<1，用来控制正负样本的权重 (通常取 0.25)
        :param gamma: 聚焦参数，gamma>0，越大越关注难分样本 (通常取 2.0)
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # 展平 tensor
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # 确保 inputs 在 0-1 之间 (如果模型输出是 logits，需要加 sigmoid；如果是概率则不需要)
        # 建议加上 eps 防止 log(0)
        epsilon = 1e-6
        # 假设输入已经是经过 sigmoid 的概率 (根据你原有 WeightedBCE 代码推断)
        # 如果训练发现 unstable，可在这里加上 inputs = torch.clamp(inputs, epsilon, 1. - epsilon)

        BCE = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE)  # pt 是模型对于该类的预测概率

        # Focal Loss 公式
        loss = self.alpha * (1 - pt) ** self.gamma * BCE

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, smooth=1.0):
        """
        Tversky Loss 用于显式控制 FP 和 FN 的权衡
        :param alpha: False Positive (误检) 的权重
        :param beta: False Negative (漏检) 的权重.
                     对于 MA 分割，为了提高 Recall，建议 beta > alpha (如 beta=0.7)
        :param smooth: 平滑项，防止除零
        """
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, inputs, targets):
        # 展平
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # 计算 TP, FP, FN
        # inputs 是预测概率，targets 是 0/1 标签
        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()

        # Tversky Index
        Tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)

        return 1 - Tversky


class FocalTverskyLoss(nn.Module):
    def __init__(self, focal_weight=0.5, tversky_weight=0.5):
        """
        复合损失函数：Focal Loss (像素级难样本挖掘) + Tversky Loss (区域级 Recall 优化)
        """
        super(FocalTverskyLoss, self).__init__()
        # 初始化两个子损失
        # 针对 MA 小目标，alpha 设为 0.25 (降低背景权重), gamma 设为 2.0 (关注难样本)
        self.focal = FocalLoss(alpha=0.25, gamma=2.0)

        # 针对 MA 漏检问题，beta 设为 0.7 (严惩漏检), alpha 设为 0.3
        self.tversky = TverskyLoss(alpha=0.4, beta=0.6)

        self.focal_weight = focal_weight
        self.tversky_weight = tversky_weight

    def forward(self, inputs, targets):
        # 计算两个损失
        f_loss = self.focal(inputs, targets)
        t_loss = self.tversky(inputs, targets)

        # 加权求和
        total_loss = self.focal_weight * f_loss + self.tversky_weight * t_loss

        return total_loss


class AdaptiveHybridLoss(nn.Module):
    """
    分段混合损失函数：解决MA分割前期训练不稳定问题

    参数说明：
    - warmup_epochs: BCE主导阶段轮数（建议5-8）
    - transition_epochs: 过渡阶段轮数（建议10-15）
    - dice_weight: Dice损失权重（默认1.0）
    - bce_weight: BCE损失权重（默认0.5）
    - focal_weight: FocalLoss权重（默认0.4）
    - tversky_weight: TverskyLoss权重（默认0.6）
    """

    def __init__(self, warmup_epochs=5, transition_epochs=10,
                 dice_weight=1.0, bce_weight=0.5,
                 focal_weight=0.4, tversky_weight=0.6):
        super(AdaptiveHybridLoss, self).__init__()

        # 初始化两个基础损失
        self.dice_bce_loss = WeightedDiceBCE(
            dice_weight=dice_weight,
            BCE_weight=bce_weight
        )
        self.focal_tversky_loss = FocalTverskyLoss(
            focal_weight=focal_weight,
            tversky_weight=tversky_weight
        )

        # 分段训练参数
        self.warmup_epochs = warmup_epochs
        self.transition_epochs = transition_epochs
        self.current_epoch = 0  # 由外部设置

    def set_epoch(self, epoch):
        """设置当前epoch（关键！）"""
        self.current_epoch = epoch

    def get_current_weight(self):
        """根据当前epoch计算混合权重"""
        if self.current_epoch < self.warmup_epochs:
            # 阶段1：纯WeightedDiceBCE
            bce_weight = 1.0
            ftl_weight = 0.0
        elif self.current_epoch < self.warmup_epochs + self.transition_epochs:
            # 阶段2：线性过渡
            progress = (self.current_epoch - self.warmup_epochs) / self.transition_epochs
            bce_weight = 1.0 - progress
            ftl_weight = progress
        else:
            # 阶段3：纯FocalTverskyLoss
            bce_weight = 0.0
            ftl_weight = 1.0

        return bce_weight, ftl_weight

    def forward(self, inputs, targets):
        # 获取当前权重
        bce_weight, ftl_weight = self.get_current_weight()

        # 计算加权损失
        if bce_weight > 0:
            dice_bce = self.dice_bce_loss(inputs, targets)
        else:
            dice_bce = torch.tensor(0.0).to(inputs.device)

        if ftl_weight > 0:
            focal_tversky = self.focal_tversky_loss(inputs, targets)
        else:
            focal_tversky = torch.tensor(0.0).to(inputs.device)

        # 混合损失
        total_loss = bce_weight * dice_bce + ftl_weight * focal_tversky

        return total_loss