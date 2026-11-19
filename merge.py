# coding=utf-8
"""
============================================================
merge_cv_results.py —— 合并多次训练的各折结果，输出最终汇总表
============================================================
适用场景：
    5 折训练中某一折崩了（如 Dice=0），单独重训了该折，
    现在把"原来的 4 折 + 重训的 1 折"合并成一张完整的 mean ± std 表。

用法：
    1. 把下方 FOLD_FILES 里 5 个折号对应的权重文件路径填好
       （重训的那折指向新时间戳文件夹，其余指向原文件夹）
    2. python merge_cv_results.py
"""
import torch
import numpy as np

## ============ 配置：每折用哪个权重文件（改成你的实际路径）============
FOLD_FILES = {
    1: '/root/rivermind-data/results_EX/EX/CAUNet/cv_models/CV_10.27_08h25/fold1_best.pth.tar',
    2: '/root/rivermind-data/results_EX/EX/CAUNet/cv_models/CV_10.27_08h25/fold2_best.pth.tar',
    3: '/root/rivermind-data/results_EX/EX/CAUNet/cv_models/CV_10.27_12h17/fold3_best.pth.tar',
    4: '/root/rivermind-data/results_EX/EX/CAUNet/cv_models/CV_10.27_08h25/fold4_best.pth.tar',
    5: '/root/rivermind-data/results_EX/EX/CAUNet/cv_models/CV_10.27_08h25/fold5_best.pth.tar',
}

METRICS = ['auc', 'aupr', 'iou', 'f1', 'dice']


def main():
    fold_results = []
    print('正在读取各折权重中的验证指标 (val_metrics)...')
    for fold in sorted(FOLD_FILES.keys()):
        path = FOLD_FILES[fold]
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        m = ckpt['val_metrics']
        fold_results.append(m)
        print('  折 {} <- {}'.format(fold, path))
        print('      epoch {} | AUPR {:.4f} Dice {:.4f}(thr~{:.2f}) IoU {:.4f} F1 {:.4f} AUC {:.4f}'.format(
            m.get('epoch', -1), m['aupr'], m['dice'], m.get('dice_t', float('nan')),
            m['iou'], m['f1'], m['auc']))

    print('\n' + '=' * 60)
    print('>>> 合并后的 5 折交叉验证结果 (mean ± std)')
    print('=' * 60)
    print('{:<8}{:<14}{:<14}{:<14}'.format('指标', '均值', '偏差值(std)', '各折数值'))
    for name in METRICS:
        vals = np.array([r[name] for r in fold_results], dtype=np.float64)
        print('{:<8}{:<14.4f}{:<14.4f}  {}'.format(
            name, vals.mean(), vals.std(ddof=1), np.round(vals, 4)))


if __name__ == '__main__':
    main()