import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_auc_score
import torch
import logging


def to_var(x, device):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    return x


def to_numpy(x):
    if not (isinstance(x, np.ndarray) or x is None):
        if x.is_cuda:
            x = x.data.cpu()
        x = x.numpy()
    return x


def logger_config(log_path):
    loggerr = logging.getLogger()
    # 输出DEBUG及以上级别的信息，针对所有输出的第一层过滤
    loggerr.setLevel(level=logging.INFO)
    # 获取文件日志句柄并设置日志级别，第二层过滤
    handler = logging.FileHandler(log_path, encoding='UTF-8')
    handler.setLevel(logging.INFO)
    # 生成并设置文件日志格式
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    # console相当于控制台输出，handler文件输出。获取流句柄并设置日志级别，第二层过滤
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # 为logger对象添加句柄
    loggerr.addHandler(handler)
    loggerr.addHandler(console)
    return loggerr


def AUPR(mask, prediction):
    threshold_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    precisions = []
    recalls = []

    verbose = False  # 修改点：将原先打印详细阈值参数的过程设置为 False 以避免冗余打印

    if verbose:
        print("阈值\tTP\tFP\tFN\tPrecision\tRecall")
        print("-" * 50)

    for thresh in threshold_list:
        thresh_pred = np.zeros(prediction.shape)
        thresh_pred[prediction >= thresh] = 1

        P = np.count_nonzero(mask)
        TP = np.count_nonzero(mask * thresh_pred)
        FP = np.count_nonzero(thresh_pred - (mask * thresh_pred))
        FN = np.count_nonzero(mask - (mask * thresh_pred))

        if (P > 0) and (TP + FP > 0):  # avoid division by 0
            precision = TP * 1.0 / (TP + FP)
            recall = TP * 1.0 / P
        else:
            precision = 1
            recall = 0

        if verbose:
            print(
                f"AUPR阈值={thresh:.2f}: TP_AUPR={TP:4d}, FP_AUPR={FP:4d}, FN_AUPR={FN:4d}, Precision={precision:.4f}, Recall={recall:.4f}")

        precisions.append(precision)
        recalls.append(recall)

    precisions.append(1)
    recalls.append(0)

    aupr_score = auc(recalls, precisions)

    if verbose:
        print("-" * 50)
        print(f"AUPR Score: {aupr_score:.6f}")

    return aupr_score


def aupr_on_batch(masks, pred):
    auprs = []
    for i in range(pred.shape[0]):
        prediction = pred[i][0].cpu().detach().numpy()
        mask = masks[i].cpu().detach().numpy()
        auprs.append(AUPR(mask, prediction))
    return np.mean(auprs)


def auc_on_batch(masks, pred):
    aucs = []
    for i in range(pred.shape[0]):
        prediction = pred[i][0].cpu().detach().numpy().reshape(-1)
        mask = masks[i].cpu().detach().numpy().reshape(-1)
        aucs.append(roc_auc_score(mask, prediction))
    return np.mean(aucs)


def IOU(mask, prediction, threshold=0.9):
    thresh_pred = np.zeros(prediction.shape)
    thresh_pred[prediction >= threshold] = 1

    TP = np.count_nonzero(mask * thresh_pred)
    FP = np.count_nonzero(thresh_pred - (mask * thresh_pred))
    FN = np.count_nonzero(mask - (mask * thresh_pred))

    # 修改点：除杂打印注释
    # print("TP_IOU={}, FP_IOU={}, FN_IOU={}".format(TP, FP, FN))

    if (TP + FP + FN) > 0:
        iou = TP * 1.0 / (TP + FP + FN)
    else:
        iou = 1.0
    return iou


def iou_on_batch(masks, pred, threshold=0.9):
    ious = []
    for i in range(pred.shape[0]):
        prediction = pred[i][0].cpu().detach().numpy()
        mask = masks[i].cpu().detach().numpy()
        ious.append(IOU(mask, prediction, threshold))
    return np.mean(ious)


def F1Score(mask, prediction, threshold=0.9):
    thresh_pred = np.zeros(prediction.shape)
    thresh_pred[prediction >= threshold] = 1

    TP = np.count_nonzero(mask * thresh_pred)
    FP = np.count_nonzero(thresh_pred - (mask * thresh_pred))
    FN = np.count_nonzero(mask - (mask * thresh_pred))

    # 修改点：除杂打印注释
    # print("TP_F1={}, FP_F1={}, FN_F1={}".format(TP, FP, FN))

    if (2 * TP + FP + FN) > 0:
        f1 = 2 * TP * 1.0 / (2 * TP + FP + FN)
    else:
        f1 = 1.0
    return f1


def f1score_on_batch(masks, pred, threshold=0.9):
    f1s = []
    for i in range(pred.shape[0]):
        prediction = pred[i][0].cpu().detach().numpy()
        mask = masks[i].cpu().detach().numpy()
        f1s.append(F1Score(mask, prediction, threshold))
    return np.mean(f1s)


def dice_on_batch(masks, pred, threshold=0.9):
    return f1score_on_batch(masks, pred, threshold)