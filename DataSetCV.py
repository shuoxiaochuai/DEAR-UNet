# coding=utf-8
"""
============================================================
DataSetCV.py —— 5折交叉验证专用数据集（新增文件，不改原代码）
============================================================
核心思想：
    你的数据物理上分成 train_images / test_images 两个文件夹，
    但我们不做任何移动/合并，只是在代码里把两个文件夹"合并成一个
    总样本池"（81张IDRiD图像），然后用 KFold 切成5份。
    每个样本记住自己来自哪个文件夹（image_sub / mask_sub），
    所以读取掩模时自动去对应的 train_masks 或 test_masks 里找。

    [适配 IDRiD 官方命名] 掩模文件名形如 IDRiD_001_EX.tif，
    即 = 图片基础名 + '_' + 任务名 + '.tif'。
    若找不到带后缀的，会兜底尝试不带后缀的 IDRiD_001.tif。
"""
import os
import numpy as np
import SimpleITK as sitk
import torchvision
from torch.utils.data import Dataset, DataLoader
from Transforms import (Resize, RandomRotate90, ToTensor, ImageEnhencer,
                        ApplyCLAHE, RandomCrop, CenterCrop)

IMG_EXTS = ('.jpg', '.jpeg', '.png')


def load_sitk(path):
    return sitk.GetArrayFromImage(sitk.ReadImage(path))


def _find_mask_path(root_dir, mask_sub, task, base):
    """
    先找  base_task.tif (如 IDRiD_001_EX.tif)，找不到再兜底 base.tif。
    返回路径或 None。
    """
    d = os.path.join(root_dir, mask_sub, task)
    p1 = os.path.join(d, base + '_' + task + '.tif')
    if os.path.exists(p1):
        return p1
    p2 = os.path.join(d, base + '.tif')
    if os.path.exists(p2):
        return p2
    return None


def _load_mask(root_dir, mask_sub, task, base):
    """读取某个任务的掩模，并归一化到 [0,1]"""
    mask_path = _find_mask_path(root_dir, mask_sub, task, base)
    if mask_path is None:
        return None
    mask = load_sitk(mask_path)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    mask = mask / 255.0
    return mask.astype(np.float32)


def build_all_samples(root_dir, tasks):
    """
    扫描 train_images 和 test_images，合并成总样本池（不改动磁盘上的任何文件）。

    过滤规则与原 DataSet.py 保持一致：
        每个任务的掩模必须存在，且不全黑（没有病灶的图不参与训练/验证）。

    返回:
        samples = [(basename, image_sub, mask_sub), ...]
        例如 ('IDRiD_001', 'train_images', 'train_masks')
    """
    samples = []
    for image_sub, mask_sub in [('train_images', 'train_masks'),
                                ('test_images', 'test_masks')]:
        img_dir = os.path.join(root_dir, image_sub)
        if not os.path.exists(img_dir):
            print('[警告] 目录不存在，已跳过: {}'.format(img_dir))
            continue
        for f in sorted(os.listdir(img_dir)):
            if not f.lower().endswith(IMG_EXTS):
                continue
            base = os.path.splitext(f)[0]
            ok = True
            for task in tasks:
                m = _load_mask(root_dir, mask_sub, task, base)
                if m is None or np.max(m) <= 0:
                    ok = False
                    break
            if ok:
                samples.append((base, image_sub, mask_sub))

    print('[样本池] 共 {} 张有效图像 (train_images + test_images 合并)'.format(len(samples)))
    return samples


class IDRiDDatasetCV(Dataset):
    """
    交叉验证版 Dataset。
    与原版 IDRiDDataset 的区别：
        原版写死了去 train_images/train_masks 找数据；
        这里每张图自己带着 image_sub / mask_sub，去自己该去的文件夹读。
    """

    def __init__(self, root_dir, samples, tasks, transform=None, green=True):
        self.root_dir = root_dir
        self.samples = samples      # [(base, image_sub, mask_sub), ...]
        self.tasks = tasks
        self.transform = transform
        self.green = green

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        base, image_sub, mask_sub = self.samples[idx]

        # ---------- 加载原始图像 ----------
        img_path = None
        for ext in IMG_EXTS:
            p = os.path.join(self.root_dir, image_sub, base + ext)
            if os.path.exists(p):
                img_path = p
                break
        if img_path is None:
            raise FileNotFoundError('Image for {} not found'.format(base))
        image = load_sitk(img_path)

        # ---------- 加载所有任务的掩模 ----------
        masks = []
        for task in self.tasks:
            m = _load_mask(self.root_dir, mask_sub, task, base)
            if m is None:
                raise FileNotFoundError('Mask for {} task {} not found'.format(base, task))
            masks.append(m)
        masks = np.stack(masks, axis=0)   # (num_tasks, H, W)

        sample = {'image': image, 'masks': masks}
        if self.transform:
            sample = self.transform(sample)
        return sample


def get_fold_loaders(root_dir, samples, train_idx, val_idx, tasks,
                     batch_size=1, green=True):
    """
    给定 KFold 切出来的 train_idx / val_idx，构建本折的 DataLoader。
    训练折用数据增强，验证折只做必要预处理（和原代码逻辑一致）。
    """
    train_samples = [samples[i] for i in train_idx]
    val_samples = [samples[i] for i in val_idx]
    print('[本折划分] 训练 {} 张 | 验证 {} 张'.format(len(train_samples), len(val_samples)))

    transform_train = torchvision.transforms.Compose([
        Resize(656),
        RandomCrop(640),
        RandomRotate90(),
        ImageEnhencer(color_jitter=True, green=False),
        ApplyCLAHE(green=green),
        ToTensor(green=green),
    ])
    transform_val = torchvision.transforms.Compose([
        Resize(642),
        CenterCrop(640),
        ApplyCLAHE(green=green),
        ToTensor(green=green),
    ])

    train_ds = IDRiDDatasetCV(root_dir, train_samples, tasks, transform_train, green)
    val_ds = IDRiDDatasetCV(root_dir, val_samples, tasks, transform_val, green)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    return train_loader, val_loader