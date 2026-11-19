# diagnose_by_aupr.py
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from DataSet import load_train_val_data
from nets.CAUNet import CAUNet
from utils_YWJ import dice_on_batch, iou_on_batch, aupr_on_batch
import warnings
warnings.filterwarnings("ignore")

def load_original_image(data_path, filename, target_shape=None):
    """
    根据 filename 从 data_path 中递归查找真实原图。
    返回 RGB numpy array, shape: H x W x 3

    target_shape: mask 的尺寸，例如 (H, W)。如果原图尺寸不同，则 resize 到该尺寸。
    """
    from PIL import Image

    # filename 可能是 list 中的字符串，也可能带路径
    filename = str(filename)

    # 去掉可能的路径，只保留文件名
    base_name = os.path.basename(filename)
    name_no_ext = os.path.splitext(base_name)[0]

    valid_exts = ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp']

    candidates = []

    # 递归查找同名图像
    for root, dirs, files in os.walk(data_path):
        root_lower = root.lower()

        # 尽量避开 mask / label / groundtruth 文件夹
        # 防止误把标注图当成原图
        if any(x in root_lower for x in ['mask', 'masks', 'label', 'labels', 'gt', 'groundtruth', 'ground_truth']):
            continue

        for f in files:
            f_lower = f.lower()
            f_no_ext = os.path.splitext(f)[0]

            if os.path.splitext(f_lower)[1] not in valid_exts:
                continue

            # 完全同名，或者去扩展名后同名
            if f == base_name or f_no_ext == name_no_ext:
                candidates.append(os.path.join(root, f))

    if len(candidates) == 0:
        raise FileNotFoundError(
            f"Cannot find original image for filename={filename} under data_path={data_path}"
        )

    # 如果找到多个，优先选择路径中包含 image/original/img 的
    candidates_sorted = sorted(
        candidates,
        key=lambda p: (
            0 if any(k in p.lower() for k in ['image', 'images', 'original', 'orig', 'img']) else 1,
            len(p)
        )
    )

    img_path = candidates_sorted[0]

    img = Image.open(img_path).convert('RGB')

    if target_shape is not None:
        h, w = target_shape
        if img.size != (w, h):
            img = img.resize((w, h), Image.BILINEAR)

    img = np.array(img)

    return img

def compute_ma_features(mask, image):
    """
    mask: 2D numpy array, 0/1
    image: 2D or 3D numpy array，真实原图，取绿色通道
    返回特征字典

    avg_contrast: 使用全局对比度（MA 区域均值 - 全图背景均值）
    """
    mask_bin = (mask > 0.5).astype(np.uint8)

    from skimage import measure
    labels = measure.label(mask_bin, connectivity=1)
    props = measure.regionprops(labels)

    num_lesions = len(props)

    if num_lesions == 0:
        return {
            'num_lesions': 0,
            'total_area': 0,
            'avg_area': 0,
            'avg_distance': 0,
            'avg_contrast': 0,
            'avg_dist_to_edge': 0,
        }

    total_area = np.sum(mask_bin)
    avg_area = total_area / num_lesions

    centroids = [prop.centroid for prop in props]
    from scipy.spatial.distance import pdist
    if len(centroids) >= 2:
        avg_distance = np.mean(pdist(centroids))
    else:
        avg_distance = 0

    # 取真实原图绿色通道
    if len(image.shape) == 3:
        img_gray = image[:, :, 1] if image.shape[2] == 3 else image[:, :, 0]
    else:
        img_gray = image
    img_gray = img_gray.astype(np.float32)

    # ===============================
    # 全局对比度计算（原版方法）
    # ===============================
    ma_pixels = img_gray[mask_bin > 0]
    bg_pixels = img_gray[mask_bin == 0]
    if len(ma_pixels) == 0 or len(bg_pixels) == 0:
        avg_contrast = 0
    else:
        ma_mean = np.mean(ma_pixels)
        bg_mean = np.mean(bg_pixels)
        avg_contrast = ma_mean - bg_mean   # 负值表示 MA 更暗

    # 到图像边界的距离（不变）
    h, w = mask_bin.shape
    coords = np.argwhere(mask_bin > 0)
    if len(coords) == 0:
        avg_dist_to_edge = 0
    else:
        dist_to_top = coords[:, 0]
        dist_to_bottom = h - 1 - coords[:, 0]
        dist_to_left = coords[:, 1]
        dist_to_right = w - 1 - coords[:, 1]
        min_dist_to_edge = np.minimum.reduce([dist_to_top, dist_to_bottom, dist_to_left, dist_to_right])
        avg_dist_to_edge = np.mean(min_dist_to_edge)

    return {
        'num_lesions': num_lesions,
        'total_area': total_area,
        'avg_area': avg_area,
        'avg_distance': avg_distance,
        'avg_contrast': avg_contrast,
        'avg_dist_to_edge': avg_dist_to_edge,
    }

def diagnose_by_aupr(model, val_loader, device, output_dir='diagnosis_aupr', data_path=None):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    all_records = []

    with torch.no_grad():
        for i, sample in enumerate(val_loader):
            images = sample['image'].cuda().float()
            masks = sample['masks'].cuda().float()
            filenames = sample['filename']

            output, _, _, _ = model(images)
            # 注意：CAUNet 输出已带 sigmoid，无需额外处理
            pred = output

            # 计算 AUPR 和 Dice
            aupr = aupr_on_batch(masks, pred)      # 使用已有函数
            dice = dice_on_batch(masks, pred)      # 使用已有函数（内部阈值0.9）
            iou = iou_on_batch(masks, pred)

            # 转换为 numpy 用于特征提取和可视化
            pred_np = pred.cpu().numpy()
            mask_np = masks.cpu().numpy()
            img_np = images.cpu().numpy()

            for b in range(images.shape[0]):
                pred_b = pred_np[b, 0]
                mask_b = mask_np[b, 0]

                # 原来的预处理后图像，保留用于可视化，不改其他逻辑
                img_b = img_np[b, 0] if img_np.shape[1] == 1 else img_np[b, 1]

                # 读取真实原图，用于计算局部对比度
                if data_path is not None:
                    orig_img_b = load_original_image(
                        data_path=data_path,
                        filename=filenames[b],
                        target_shape=mask_b.shape
                    )
                else:
                    # 如果没有传 data_path，则退回使用当前 img_b
                    # 正常情况下 main 中会传入 data_path
                    orig_img_b = img_b

                # 二值化预测（用于 FP/FN 统计，使用阈值 0.985 以保持一致）
                pred_bin = (pred_b > 0.985).astype(np.uint8)
                fp = np.sum((pred_bin == 1) & (mask_b == 0))
                fn = np.sum((pred_bin == 0) & (mask_b == 1))

                # 这里改为使用真实原图 orig_img_b
                ma_feat = compute_ma_features(mask_b, orig_img_b)

                record = {
                    'filename': filenames[b],
                    'aupr': aupr,
                    'dice': dice,
                    'iou': iou,
                    'fp': fp,
                    'fn': fn,
                    **ma_feat
                }
                all_records.append(record)

                # 可选：保存 AUPR 低的图片的可视化
                if aupr < 0.45:   # 与分组 bad 阈值一致
                    plt.figure(figsize=(12, 4))
                    plt.subplot(131)
                    plt.imshow(img_b, cmap='gray')
                    plt.title('Original')
                    plt.axis('off')
                    plt.subplot(132)
                    plt.imshow(mask_b, cmap='hot')
                    plt.title('Ground Truth')
                    plt.axis('off')
                    plt.subplot(133)
                    plt.imshow(pred_b, cmap='hot')
                    plt.title('Prediction (prob)')
                    plt.axis('off')
                    plt.savefig(os.path.join(output_dir, f'bad_aupr_{filenames[b]}_aupr{aupr:.3f}.png'))
                    plt.close()

    import pandas as pd
    df = pd.DataFrame(all_records)

    # 输出平均 AUPR
    avg_aupr = df['aupr'].mean()
    print(f"Average AUPR over all {len(df)} images: {avg_aupr:.4f}")

    # 按 AUPR 分组
    good = df[df['aupr'] > 0.5]
    bad = df[df['aupr'] < 0.45]
    print(f"Total samples: {len(df)}")
    print(f"Good (AUPR > 0.5): {len(good)}")
    print(f"Bad (AUPR < 0.45): {len(bad)}")

    # 统计特征差异
    features = ['num_lesions', 'total_area', 'avg_area', 'avg_distance', 'avg_contrast', 'avg_dist_to_edge']
    for feat in features:
        good_vals = good[feat].dropna()
        bad_vals = bad[feat].dropna()
        if len(good_vals) > 0 and len(bad_vals) > 0:
            stat, p = stats.mannwhitneyu(good_vals, bad_vals, alternative='two-sided')
            print(f"{feat}: good mean={good_vals.mean():.2f}, bad mean={bad_vals.mean():.2f}, p={p:.4f}")
        else:
            print(f"{feat}: insufficient data")

    # 保存详细记录
    df.to_csv(os.path.join(output_dir, 'diagnosis_aupr.csv'), index=False)

    # 箱线图
    for feat in features:
        plt.figure()
        data_to_plot = [good[feat].dropna(), bad[feat].dropna()]
        if len(data_to_plot[0]) > 0 and len(data_to_plot[1]) > 0:
            plt.boxplot(data_to_plot, labels=['Good AUPR>0.5', 'Bad AUPR<0.45'])
            plt.title(f'{feat} distribution (grouped by AUPR)')
            plt.ylabel(feat)
            plt.savefig(os.path.join(output_dir, f'boxplot_aupr_{feat}.png'))
            plt.close()

    print(f"Diagnosis completed. Results saved in {output_dir}")
    return df

if __name__ == '__main__':
    data_path = '/root/autodl-tmp/Segmentation_dataset'
    model_path = '/root/autodl-tmp/results_training/MA/CAUNet/save/models/Test_06.01_18h00/best_aupr.pth.tar'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    _, val_loader = load_train_val_data(tasks=['MA'], data_path=data_path, batch_size=1, green=True)

    model = CAUNet().to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])

    diagnose_by_aupr(
        model,
        val_loader,
        device,
        output_dir='/root/autodl-tmp/diagnose_aupr_61',
        data_path=data_path
    )