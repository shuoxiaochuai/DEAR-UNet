# coding=utf-8
"""
============================================================
main_CV.py —— 5折交叉验证训练主程序（不改原代码）
============================================================
流程：
    1. 把 train_images + test_images 合并成 81 张总样本池
    2. KFold(n_splits=5) 切成5份
    3. 每一折：用全新的模型从头训练 epochs 轮，
       每轮在"本折验证集"上验证，记录本折最优 AUPR 时的各项指标
    4. 5折全部跑完后，打印每个指标的 均值 ± 标准差（偏差值）

[本次更新] 验证时的 Dice 不再固定阈值 0.9，
    而是对每张图遍历阈值 [0.8, 0.9, 0.95, 0.99] 取最大值计入，
    并在 epoch 汇总行里打印最优阈值（thr≈x.xx）。
    注意：最优模型仍按 AUPR 挑选，Dice 只是展示口径变宽。
"""

import os
import time
import numpy as np
import torch
from sklearn.model_selection import KFold

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

from DataSetCV import build_all_samples, get_fold_loaders
from nets.CAUNet import CAUNet
from Losses import AdaptiveHybridLoss, LBTW_Loss, LBTW_algorithm
from LR_methods import CosineAnnealingWarmRestarts
from utils_YWJ import (auc_on_batch, aupr_on_batch, iou_on_batch,
                       f1score_on_batch, dice_on_batch)

## ============ 超参数（与原 main_YWJ.py 保持一致）============
learning_rate = 5e-4
epochs = 30
batch_size = 1
n_splits = 5          # 折数
random_state = 42     # 随机种子，保证每次切分完全一样，结果可复现
green = True
task_name = 'EX'      # <<< 改成你要训练的任务: MA / HE / EX / SE / OD

## ============ Dice 多阈值遍历配置 ============
DICE_THRESHOLDS = [0.8, 0.9, 0.95, 0.99]   # 每张图在这些阈值中取最大 Dice

## ============ 折选择 ============
# 正常 5 折训练：保持 [] 不动
# 只重训某几折：填折号列表，如 [3] 或 [2, 4]（折号从 1 开始）
# 重训结果会存进新的时间戳文件夹，不影响已有结果
RETRAIN_FOLDS = [3]

## ============ 路径（改成你自己的）============
data_path = '/root/rivermind-data/Segmentation_dataset3'
output_base = '/root/rivermind-data/results_EX'


def run_one_epoch(loader, model, criterion, optimizer, lbtw, epoch_idx):
    """跑一个 epoch，返回各项指标的均值（loss/auc/aupr/iou/f1/dice/最优阈值）"""
    mode = 'Train' if model.training else 'Val'
    sums = {'loss': 0., 'auc': 0., 'aupr': 0., 'iou': 0., 'f1': 0., 'dice': 0., 'dice_t': 0.}
    n = 0
    for i, sample in enumerate(loader, 1):
        images = sample['image'].cuda().float()
        masks = sample['masks'].cuda().float()

        output, pred2, pred3, pred1 = model(images)
        # 注意顺序：criterion(out, comb3, comb2, comb1, target)
        _, out_loss, c3_loss, c2_loss, c1_loss = criterion(output, pred3, pred2, pred1, masks)

        if model.training:
            loss, *_ = lbtw(i, out_loss, c3_loss, c2_loss, c1_loss)
            if torch.isnan(loss):          # [新增] NaN 保护：跳过该 batch
                print('   [警告] epoch {} batch {} loss 为 NaN，已跳过'.format(epoch_idx + 1, i))
                del loss
                torch.cuda.empty_cache()
                continue
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)  # [新增] 梯度裁剪防爆炸
            optimizer.step()

        bs = images.size(0)
        sums['loss'] += out_loss.item() * bs
        sums['auc'] += auc_on_batch(masks, output) * bs
        sums['aupr'] += aupr_on_batch(masks, output) * bs
        sums['iou'] += iou_on_batch(masks, output) * bs
        sums['f1'] += f1score_on_batch(masks, output) * bs

        # [多阈值遍历] 每张图在 DICE_THRESHOLDS 中取最大 Dice
        dice_candidates = [dice_on_batch(masks, output, threshold=t) for t in DICE_THRESHOLDS]
        best_idx = int(np.argmax(dice_candidates))
        sums['dice'] += dice_candidates[best_idx] * bs
        sums['dice_t'] += DICE_THRESHOLDS[best_idx] * bs   # 最优阈值，仅用于显示

        n += bs

        if i % 20 == 0:
            print('   [{}] epoch {} batch {}/{}  loss={:.4f}'.format(
                mode, epoch_idx + 1, i, len(loader), out_loss.item()))
        torch.cuda.empty_cache()

    return {k: v / max(n, 1) for k, v in sums.items()}


def train_one_fold(fold, train_loader, val_loader):
    """训练一折：模型、优化器、LBTW、scheduler 全部从头新建"""
    model = CAUNet().cuda()

    # 分层学习率（与原代码一致）：STBM 用更低学习率
    stbm_params = [p for name, p in model.named_parameters() if 'stbm' in name]
    other_params = [p for name, p in model.named_parameters() if 'stbm' not in name]
    optimizer = torch.optim.Adam([
        {'params': other_params, 'lr': learning_rate},
        {'params': stbm_params, 'lr': learning_rate * 0.1},
    ])
    lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=epochs, eta_min=1e-6)
    lbtw = LBTW_algorithm()
    criterion = LBTW_Loss(AdaptiveHybridLoss())

    best = {'aupr': -1.0}
    best_state = None
    for epoch in range(epochs):
        criterion.criteria.set_epoch(epoch)   # 告诉 AdaptiveHybridLoss 现在是第几轮

        model.train(True)
        tr = run_one_epoch(train_loader, model, criterion, optimizer, lbtw, epoch)

        model.eval()
        with torch.no_grad():
            va = run_one_epoch(val_loader, model, criterion, optimizer, lbtw, epoch)

        lr_scheduler.step()
        print('[Fold {}] Epoch {}/{} | train loss {:.4f} aupr {:.4f} | '
              'val AUPR {:.4f} Dice {:.4f}(thr~{:.2f}) IoU {:.4f} F1 {:.4f} AUC {:.4f}'.format(
                  fold + 1, epoch + 1, epochs, tr['loss'], tr['aupr'],
                  va['aupr'], va['dice'], va['dice_t'], va['iou'], va['f1'], va['auc']))

        # 以本折验证集 AUPR 最优为准，保存该轮的完整指标和权重
        if va['aupr'] > best['aupr']:
            best = dict(va)
            best['epoch'] = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    return best, best_state


def main():
    t_start = time.time()
    samples = build_all_samples(data_path, [task_name])
    assert len(samples) >= n_splits, '有效样本数少于折数，请检查数据和任务名'

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    # [新增] 按当前时间建会话文件夹，多次训练互不覆盖
    session_name = 'CV_' + time.strftime('%m.%d_%Hh%M')
    save_dir = os.path.join(output_base, task_name, 'CAUNet', 'cv_models', session_name)
    os.makedirs(save_dir, exist_ok=True)
    print('本次训练结果保存到: {}'.format(save_dir))

    fold_results = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(samples)):
        if RETRAIN_FOLDS and (fold + 1) not in RETRAIN_FOLDS:
            print('>>> 跳过第 {} 折（RETRAIN_FOLDS 未包含）'.format(fold + 1))
            continue
        print('\n' + '=' * 60)
        print('>>> 开始第 {}/{} 折'.format(fold + 1, n_splits))
        print('=' * 60)

        train_loader, val_loader = get_fold_loaders(
            data_path, samples, train_idx, val_idx, [task_name], batch_size, green)

        best, best_state = train_one_fold(fold, train_loader, val_loader)

        # 保存本折最优模型
        fold_save = os.path.join(save_dir, 'fold{}_best.pth.tar'.format(fold + 1))
        torch.save({'fold': fold + 1, 'state_dict': best_state,
                    'val_metrics': best}, fold_save)

        fold_results.append(best)
        print('<<< 第 {} 折完成 | 最优轮次 epoch {} | AUPR {:.4f} Dice {:.4f}(thr~{:.2f}) '
              'IoU {:.4f} F1 {:.4f} AUC {:.4f}'.format(
                  fold + 1, best['epoch'], best['aupr'], best['dice'], best['dice_t'],
                  best['iou'], best['f1'], best['auc']))

    # ================= 汇总：均值 ± 标准差（偏差值） =================
    if RETRAIN_FOLDS:
        print('\n[提示] 本次只重训了 {} 折，完整 5 折汇总请用 merge_cv_results.py 合并各次结果'.format(RETRAIN_FOLDS))
        print('本次重训各折指标：')
    print('\n' + '=' * 60)
    print('>>> 5折交叉验证最终结果 (mean ± std)')
    print('=' * 60)
    print('{:<8}{:<14}{:<14}{:<14}'.format('指标', '均值', '偏差值(std)', '各折数值'))
    for m in ['auc', 'aupr', 'iou', 'f1', 'dice']:
        vals = np.array([r[m] for r in fold_results], dtype=np.float64)
        mean_v = vals.mean()
        std_v = vals.std(ddof=1)   # 样本标准差（5个样本，除以 n-1）
        print('{:<8}{:<14.4f}{:<14.4f}  {}'.format(
            m, mean_v, std_v, np.round(vals, 4)))

    print('\n耗时: {:.1f} 分钟'.format((time.time() - t_start) / 60.0))
    print('各折最优模型保存在: {}'.format(save_dir))


if __name__ == '__main__':
    main()