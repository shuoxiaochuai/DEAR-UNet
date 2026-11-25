# coding=utf-8
import torch
import torch.optim
from tensorboardX import SummaryWriter
import os

os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
import time
import numpy as np
from utils_YWJ import auc_on_batch, aupr_on_batch, iou_on_batch, f1score_on_batch, dice_on_batch, logger_config
from DataSet import load_train_val_data
from nets.CAUNet import CAUNet
import torch.nn as nn
from Losses import *
from LR_methods import CosineAnnealingWarmRestarts
import logging
import warnings

warnings.filterwarnings("ignore")

## PARAMETERS OF THE MODEL
use_cuda = torch.cuda.is_available()
learning_rate = 5e-4
n_labels = 1
epochs = 45
batch_size = 4
data_shape = 640
print_frequency = 1
save_frequency = 5
save_model = True
tensorboard = True
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

## Dice 多阈值自动搜索配置
DICE_THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]


def print_summary(epoch, i, nb_batch, loss, loss_name, batch_time,
                  average_loss, average_time, auc, average_auc, aupr, average_aupr,
                  dice, average_dice, best_thr, average_best_thr, mode, lr):
    '''
    控制台打印：去除 F1/IoU，Dice 显示自动搜索的最优阈值。
    '''
    summary = '   [' + str(mode) + '] Epoch: [{0}][{1}/{2}]  '.format(epoch, i, nb_batch)
    summary += '{}: {:.4f} (Avg: {:.4f}) | '.format(loss_name, loss, average_loss)
    summary += 'AUC: {:.4f} (Avg: {:.4f}) | '.format(auc, average_auc)
    summary += 'AUPR: {:.4f} (Avg: {:.4f}) | '.format(aupr, average_aupr)
    summary += 'Dice: {:.4f} (Avg: {:.4f}) thr={:.2f}(Avg thr: {:.2f}) | '.format(
        dice, average_dice, best_thr, average_best_thr)

    if mode == 'Train':
        summary += 'LR: {:.2e} | '.format(lr)

    summary += 'Time: {:.2f}s (Avg: {:.2f}s)'.format(batch_time, average_time)

    # 直接 print，每次占据独立一行
    print(summary)


def save_checkpoint(state, save_path):
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    if state['best_model']:
        filename = os.path.join(save_path, 'best_model.pth.tar')
    else:
        filename = os.path.join(save_path, 'model_ep{}.pth.tar'.format(state['epoch']))
    torch.save(state, filename)


def train_one_epoch(loader, model, criterion, optimizer, writer, epoch, lr_scheduler, model_type, lbtw_algorithm,
                    batch_size):
    logging_mode = 'Train' if model.training else 'Val'

    end = time.time()
    time_sum, loss_sum, auc_sum, aupr_sum, iou_sum, f1_sum, dice_sum, thr_sum = 0, 0, 0, 0, 0, 0, 0, 0
    auprs = []

    for (i, sample) in enumerate(loader, 1):
        images, masks = sample['image'].cuda().float(), sample['masks'].cuda().float()
        loss_name = criterion._get_name() if hasattr(criterion, '_get_name') else criterion.__name__

        output, pred_comb2, pred_comb3, pred_comb1 = model(images)
        losses, out_loss, c3_loss, c2_loss, c1_loss = criterion(output, pred_comb3, pred_comb2, pred_comb1, masks)

        if model.training:
            loss, _, _, _, _ = lbtw_algorithm(i, out_loss, c3_loss, c2_loss, c1_loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 计算所有评估指标
        train_auc = auc_on_batch(masks, output)
        train_aupr = aupr_on_batch(masks, output)
        train_iou = iou_on_batch(masks, output)
        train_f1 = f1score_on_batch(masks, output)

        # 多阈值自动搜索最优 Dice
        dice_candidates = [dice_on_batch(masks, output, threshold=t) for t in DICE_THRESHOLDS]
        best_idx = int(np.argmax(dice_candidates))
        train_dice = dice_candidates[best_idx]
        train_best_thr = DICE_THRESHOLDS[best_idx]

        auprs.append(train_aupr)

        # measure elapsed time
        batch_time = time.time() - end

        time_sum += len(images) * batch_time
        loss_sum += len(images) * out_loss.item()
        auc_sum += len(images) * train_auc
        aupr_sum += len(images) * train_aupr
        iou_sum += len(images) * train_iou
        f1_sum += len(images) * train_f1
        dice_sum += len(images) * train_dice
        thr_sum += len(images) * train_best_thr

        # 计算并保存平均值
        average_loss = loss_sum / (i * batch_size)
        average_time = time_sum / (i * batch_size)
        train_auc_average = auc_sum / (i * batch_size)
        train_aupr_avg = aupr_sum / (i * batch_size)
        train_iou_avg = iou_sum / (i * batch_size)
        train_f1_avg = f1_sum / (i * batch_size)
        train_dice_avg = dice_sum / (i * batch_size)
        train_thr_avg = thr_sum / (i * batch_size)

        end = time.time()
        torch.cuda.empty_cache()

        if i % print_frequency == 0:
            print_summary(epoch + 1, i, len(loader), out_loss.item(), loss_name, batch_time,
                          average_loss, average_time, train_auc, train_auc_average,
                          train_aupr, train_aupr_avg,
                          train_dice, train_dice_avg, train_best_thr, train_thr_avg,
                          logging_mode, lr=optimizer.param_groups[0]['lr'] if optimizer else 0.0)

        if tensorboard and writer is not None:
            step = epoch * len(loader) + i
            writer.add_scalar(logging_mode + '_loss', out_loss.item(), step)
            writer.add_scalar(logging_mode + '_aupr', train_aupr, step)
            writer.add_scalar(logging_mode + '_iou', train_iou, step)
            writer.add_scalar(logging_mode + '_f1', train_f1, step)
            writer.add_scalar(logging_mode + '_dice', train_dice, step)

    if lr_scheduler is not None:
        lr_scheduler.step()

    return average_loss, train_aupr_avg, train_iou_avg, train_f1_avg, train_dice_avg


def main_loop(data_path, batch_size=1, model_type='', green=False, tensorboard=True, task_name=None, data_shape=640):
    train_loader, val_loader = load_train_val_data(tasks=task_name, data_path=data_path, batch_size=batch_size,
                                                   green=green)

    model = CAUNet().cuda()
    base_loss = AdaptiveHybridLoss()
    criterion = LBTW_Loss(base_loss)

    # 分层学习率: STBM模块使用更低的学习率,保护预训练卷积权重
    stbm_params = []
    other_params = []
    for name, param in model.named_parameters():
        if 'stbm' in name:
            stbm_params.append(param)
        else:
            other_params.append(param)

    optimizer = torch.optim.Adam([
        {'params': other_params, 'lr': learning_rate},  # 主网络: 5e-4
        {'params': stbm_params, 'lr': learning_rate * 0.1}  # STBM: 5e-5
    ])
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=15, eta_min=1e-6)
    lbtw_algorithm = LBTW_algorithm()

    if tensorboard:
        log_dir = os.path.join(tensorboard_folder, session_name)
        logger.info('log dir: {}'.format(log_dir))
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
        writer = SummaryWriter(log_dir)
    else:
        writer = None

    # 仅跟踪 AUPR 和 Dice 的最佳值
    max_aupr = 0.0
    max_dice = 0.0

    # 确保模型保存目录存在
    os.makedirs(model_path, exist_ok=True)

    for epoch in range(epochs):
        logger.info('\n========= Epoch [{}/{}] ========='.format(epoch + 1, epochs))
        logger.info(session_name)

        criterion.criteria.set_epoch(epoch)

        # Train
        model.train(True)
        logger.info('Training with batch size : {}'.format(batch_size))
        train_loss, train_aupr, train_iou, train_f1, train_dice = train_one_epoch(
            train_loader, model, criterion, optimizer, writer, epoch, None, model_type, lbtw_algorithm, batch_size)

        # Val
        model.eval()
        logger.info('Validation : ')
        with torch.no_grad():
            val_loss, val_aupr, val_iou, val_f1, val_dice = train_one_epoch(
                val_loader, model, criterion, optimizer, writer, epoch, lr_scheduler, model_type, lbtw_algorithm,
                batch_size)

        # ================= 分别保存最佳 AUPR 和最佳 Dice 模型 =================
        # 保存最佳 AUPR 模型（epoch > 5 才保存，与原逻辑一致）
        if val_aupr > max_aupr and epoch + 1 > 5:
            max_aupr = val_aupr
            aupr_save_path = os.path.join(model_path, 'best_aupr.pth.tar')
            torch.save({
                'epoch': epoch + 1,
                'model': model_type,
                'state_dict': model.state_dict(),
                'val_aupr': val_aupr,
                'val_dice': val_dice,
                'val_iou': val_iou,
                'val_f1': val_f1,
                'optimizer': optimizer.state_dict()
            }, aupr_save_path)
            logger.info(f'Best AUPR model saved: {aupr_save_path} (AUPR: {val_aupr:.4f})')

        # 保存最佳 Dice 模型（epoch > 5 才保存）
        if val_dice > max_dice and epoch + 1 > 5:
            max_dice = val_dice
            dice_save_path = os.path.join(model_path, 'best_dice.pth.tar')
            torch.save({
                'epoch': epoch + 1,
                'model': model_type,
                'state_dict': model.state_dict(),
                'val_dice': val_dice,
                'val_aupr': val_aupr,
                'val_iou': val_iou,
                'val_f1': val_f1,
                'optimizer': optimizer.state_dict()
            }, dice_save_path)
            logger.info(f'Best Dice model saved: {dice_save_path} (Dice: {val_dice:.4f})')

        # 可选：打印当前最佳值
        logger.info(
            'Current best - AUPR: {:.4f}, Dice: {:.4f}'.format(max_aupr, max_dice)
        )

    return model

if __name__ == '__main__':
    task_name = 'MA'
    model_name = 'CAUNet'

    main_path = '/root/rivermind-data/Segmentation_dataset3'
    output_base = '/root/rivermind-data/results_202611'
    save_path = os.path.join(output_base, task_name, model_name, 'save')

    session_name = 'Test_' + time.strftime('%m.%d_%Hh%M')
    model_path = os.path.join(save_path, 'models', session_name)
    tensorboard_folder = os.path.join(save_path, 'tensorboard_logs')

    if not os.path.exists(os.path.join(save_path, 'log_file')):
        os.makedirs(os.path.join(save_path, 'log_file'))

    logger = logger_config(os.path.join(save_path, 'log_file', session_name + ".log"))

    main_loop(data_path=main_path, batch_size=batch_size, model_type=model_name, green=True, task_name=[task_name])