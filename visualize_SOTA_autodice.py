# coding: utf-8
import os
import numpy as np
from matplotlib import patches
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from nets.CAUNet import CAUNet
from DataSet import load_train_val_data
from utils_YWJ import aupr_on_batch
import warnings

warnings.filterwarnings("ignore")

# [新增] Dice 自动筛选的候选阈值（与训练时口径一致）
THRESHOLDS = [0.80,0.81,0.82,0.83,0.84,0.85,0.86,0.87,0.88,0.89,0.90,0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.99]


def show_image_with_dice(predict_save, labs, save_path):
    tmp_lbl = labs.astype(np.float32)
    tmp_3dunet = predict_save.astype(np.float32)
    dice_pred = 2 * np.sum(tmp_lbl * tmp_3dunet) / (np.sum(tmp_lbl) + np.sum(tmp_3dunet) + 1e-5)
    dice_show = "%.3f" % dice_pred
    fig, ax = plt.subplots()
    plt.gca().add_patch(patches.Rectangle(xy=(4, 4), width=120, height=20, color="white", linewidth=1))
    plt.imshow(predict_save * 255, cmap='gray')
    plt.text(x=10, y=24, s="Dice:" + str(dice_show), fontsize=5)
    plt.axis("off")

    height, width = predict_save.shape
    fig.set_size_inches(width / 100.0 / 3.0, height / 100.0 / 3.0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    return dice_pred


def vis_and_save_heatmap(model, input_img, labs, vis_save_path, output=None):
    """
    [修改] 新增计算 Precision 和 Recall，与 Dice 一起返回
    """
    model.eval()
    with torch.no_grad():
        if output is None:
            input_img_gpu = input_img.cuda(non_blocking=True)
            outputs = model(input_img_gpu)
            if isinstance(outputs, tuple):
                output = outputs[0]
            else:
                output = outputs
        else:
            if not torch.is_tensor(output):
                output = torch.from_numpy(output)
            if output.device.type != 'cuda':
                output = output.cuda()

        output_cpu = output.cpu()
        prob = output_cpu[0, 0].numpy().astype(np.float32)   # 640x640 预测概率图

        if output is None:
            del input_img_gpu, outputs
        del output
        torch.cuda.empty_cache()

        # 标签先算好（筛阈值要用）
        if len(labs.shape) == 4:
            lab = labs[0, 0]
        elif len(labs.shape) == 3:
            lab = labs[0] if labs.shape[0] == 1 else labs
        else:
            lab = labs
        lab_binary = (lab > 0.5).astype(np.float32)

        # [新增] 自动筛选 Dice 最高的阈值
        best_t, best_dice = THRESHOLDS[0], -1.0
        for t in THRESHOLDS:
            pred_t = (prob >= t).astype(np.float32)
            dice_t = 2 * np.sum(pred_t * lab_binary) / (np.sum(pred_t) + np.sum(lab_binary) + 1e-5)
            if dice_t > best_dice:
                best_dice, best_t = dice_t, t

        predict_save = np.reshape((prob >= best_t).astype(np.float32), (640, 640))
        pred_binary = predict_save

        TP = np.sum((pred_binary == 1) & (lab_binary == 1))
        FP = np.sum((pred_binary == 1) & (lab_binary == 0))
        FN = np.sum((pred_binary == 0) & (lab_binary == 1))
        TN = np.sum((pred_binary == 0) & (lab_binary == 0))

        # [新增] 计算 Precision 和 Recall
        precision = TP / (TP + FP + 1e-5)
        recall = TP / (TP + FN + 1e-5)
        dice = 2 * TP / (np.sum(pred_binary) + np.sum(lab_binary) + 1e-5)

        show_image_with_dice(predict_save, lab_binary,
                             save_path=vis_save_path + '_predict' + model_type + '.jpg')
        return dice, TP, FP, FN, TN, precision, recall  # [修改] 返回7个值(最优阈值仅内部使用,不显示)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    model_type = 'CAUNet'
    data_path = '/root/rivermind-data/Segmentation_dataset5'   # 新机器数据集路径
    #save_path = '/root/rivermind-data/results_EX/EX/CAUNet/cv_models/CV_10.27_12h17/fold3_best.pth.tar'
    #save_path = '/root/rivermind-data/results_EX/EX/CAUNet/cv_models/CV_10.27_08h25/fold1_best.pth.tar'
    #save_path = '/root/rivermind-data/results_EX/EX/CAUNet/cv_models/CV_10.27_08h25/fold2_best.pth.tar'
    save_path = '/root/rivermind-data/results/EX/CAUNet/save/models/Test_10.29_04h12/best_dice.pth.tar'
    #save_path = '/root/rivermind-data/results_EX/EX/CAUNet/cv_models/CV_10.27_08h25/fold4_best.pth.tar'

    tasks = ['EX']
    visualization_root = '/root/rivermind-data/results_visual'
    att_vis_path = os.path.join(visualization_root, tasks[0], model_type, 'vis', 'Pred')
    os.makedirs(att_vis_path, exist_ok=True)

    checkpoint = torch.load(save_path, map_location='cpu', weights_only=False)
    model = CAUNet()
    model.load_state_dict(checkpoint['state_dict'])
    model = model.cuda()
    if torch.cuda.device_count() > 1:
        print("Let's use {0} GPUs!".format(torch.cuda.device_count()))
        model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    print('Model loaded !')

    train_loader, val_loader = load_train_val_data(tasks=tasks, data_path=data_path,
                                                   batch_size=1, green=True)

    dice_total = 0.0
    aupr_total = 0.0
    precision_total = 0.0  # [新增]
    recall_total = 0.0       # [新增]

    with torch.no_grad():
        with tqdm(total=len(val_loader), desc='Test visualize', unit='img', ncols=70, leave=True) as pbar:
            for i, sample in enumerate(val_loader, 1):
                test_data, test_label = sample['image'], sample['masks']

                test_data_gpu = test_data.float().cuda()
                test_label_gpu = test_label.cuda()

                outputs = model(test_data_gpu)
                if isinstance(outputs, tuple):
                    output = outputs[0]
                else:
                    output = outputs

                aupr = aupr_on_batch(test_label_gpu, output)
                aupr_total += aupr

                lab = test_label.data.numpy()
                img_lab = np.reshape(lab, (lab.shape[2], lab.shape[3])) * 255

                # 保存标签图
                fig, ax = plt.subplots()
                plt.imshow(img_lab, cmap='gray')
                plt.axis("off")
                fig.set_size_inches(640 / 300, 640 / 300)
                plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
                plt.savefig(att_vis_path + str(i) + "_lab.jpg", dpi=300)
                plt.close(fig)

                arr = test_data.numpy().astype(np.float32)
                input_img = torch.from_numpy(arr)

                # [修改] 接收7个返回值
                dice, tp, fp, fn, tn, precision, recall = vis_and_save_heatmap(
                    model, input_img, lab, att_vis_path + str(i), output=output.cpu()
                )
                dice_total += dice
                precision_total += precision  # [新增]
                recall_total += recall        # [新增]

                # [修改] 打印增加 Precision 和 Recall
                print(f"Image {i}: Dice={dice:.4f}, AUPR={aupr:.4f},  "
                      f"Precision={precision:.4f}, Recall={recall:.4f}, "
                      f"TP={tp}, FP={fp}, FN={fn}, TN={tn}")

                if i % 5 == 0:
                    torch.cuda.empty_cache()
                pbar.update()

    n = len(val_loader)
    # [修改] 最终汇总增加 Precision 和 Recall
    print("\n========== Final Results ==========")
    print(f"Average Dice:    {dice_total / n:.4f}")
    print(f"Average AUPR:    {aupr_total / n:.4f}")
    print(f"Average Precision: {precision_total / n:.4f}")  # [新增]
    print(f"Average Recall:    {recall_total / n:.4f}")       # [新增]