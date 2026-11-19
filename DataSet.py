# coding=utf-8
import torchvision
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
import SimpleITK as sitk
from Transforms import Resize, RandomRotate90, ToTensor, ImageEnhencer, ApplyCLAHE, RandomCrop, CenterCrop


def get_files_info(data_path):
    '''
        打印数据文件夹基本信息（兼容 DDR 数据集结构）
    '''
    files = os.listdir(data_path)
    print('\t Files info')
    print('Content of the folder : {}'.format(files))
    print('Length of the folder : {}'.format(len(files)))

    tasks = ['MA', 'HE', 'EX', 'SE', 'OD']
    for task in tasks:
        task_path = os.path.join(data_path, 'train_masks/' + task + '/')
        if os.path.exists(task_path):
            task_files = os.listdir(task_path)
            print('Number of train masks for ' + task + ' task :', len(task_files))

    train_images_path = os.path.join(data_path, 'train_images')
    if os.path.exists(train_images_path):
        print('Number of train images : %s' % len(os.listdir(train_images_path)))

    test_images_path = os.path.join(data_path, 'test_images')
    if os.path.exists(test_images_path):
        print('Number of test images : %s' % len(os.listdir(test_images_path)))
    print('-' * 20)
    print('\n')


def load_sitk(path):
    return sitk.GetArrayFromImage(sitk.ReadImage(path))


def _find_mask_path(root_dir, mask_task_dir, base, task):
    """[新增] 适配 IDRiD 官方命名: 先试 base_任务.tif (如 IDRiD_001_MA.tif)，
    找不到再兜底 base.tif。"""
    p1 = os.path.join(root_dir, mask_task_dir, base + '_' + task + '.tif')
    if os.path.exists(p1):
        return p1
    p2 = os.path.join(root_dir, mask_task_dir, base + '.tif')
    if os.path.exists(p2):
        return p2
    return None


class IDRiDDataset(Dataset):
    def __init__(self, mode='train', root_dir='Segmentation.nosync/',
                 transform=None, tasks=['MA'], data_augmentation=True, green=False):
        super(IDRiDDataset, self).__init__()

        if mode == 'train':
            mask_file, image_file = 'train_masks/', 'train_images/'
        elif mode == 'val':
            mask_file, image_file = 'test_masks/', 'test_images/'
        else:
            raise EnvironmentError('Invalid mode, choose "train" or "val"')

        self.mode = mode
        self.transform = transform
        self.mask_file = mask_file
        self.image_file = image_file
        self.root_dir = root_dir
        self.tasks = tasks
        self.data_augmentation = data_augmentation

        # ---------- 扫描图像文件，提取基础名（不含扩展名） ----------
        img_dir = os.path.join(self.root_dir, self.image_file)
        if not os.path.exists(img_dir):
            raise FileNotFoundError(f"Image directory not found: {img_dir}")

        all_basenames = set()
        for f in os.listdir(img_dir):
            # 支持常见图像扩展名
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                base = os.path.splitext(f)[0]
                all_basenames.add(base)

        print(f"[{mode}] 扫描到 {len(all_basenames)} 张原始图像，正在验证掩模并剔除全黑样本...")

        # ---------- 筛选有效样本：所有任务的掩模存在且不全黑 ----------
        self.valid_basenames = []
        for base in all_basenames:
            keep = True
            for task in tasks:
                mask_path = _find_mask_path(self.root_dir, self.mask_file + task, base, task)
                if mask_path is None:
                    keep = False
                    break
                # 检查是否全黑（无病灶）
                mask = load_sitk(mask_path)
                if mask.ndim == 3:
                    mask = mask[:, :, 0]   # 取第一个通道（通常为单通道）
                if np.max(mask) <= 0:      # 全黑则丢弃
                    keep = False
                    break
            if keep:
                self.valid_basenames.append(base)

        print(f"[{mode}] 有效样本数：{len(self.valid_basenames)} / {len(all_basenames)}")

    def __len__(self):
        return len(self.valid_basenames)

    def __getitem__(self, idx):
        base = self.valid_basenames[idx]

        # ---------- 加载原始图像（尝试常见扩展名） ----------
        img_path = None
        for ext in ['.jpg', '.jpeg', '.png']:
            path = os.path.join(self.root_dir, self.image_file, base + ext)
            if os.path.exists(path):
                img_path = path
                break
        if img_path is None:
            raise FileNotFoundError(f"Image for {base} not found")
        image = load_sitk(img_path)

        # ---------- 加载所有任务的掩模 ----------
        masks = []
        for task in self.tasks:
            mask_path = _find_mask_path(self.root_dir, self.mask_file + task, base, task)
            mask = load_sitk(mask_path)
            if mask.ndim == 3:
                mask = mask[:, :, 0]
            # 归一化到 [0,1]（原值为 0 或 255）
            mask = mask / 255.0
            masks.append(mask.astype(np.float32))

        masks = np.stack(masks, axis=0)   # shape: (num_tasks, H, W)

        sample = {'image': image, 'masks': masks}
        if self.transform:
            sample = self.transform(sample)

        return sample


def load_train_val_data(tasks=['MA'], data_path='Segmentation.nosync/', batch_size=8, green=False):
    get_files_info(data_path)

    # 训练数据增强
    transforms_list = [
        Resize(656),
        RandomCrop(640),
        RandomRotate90(),
        ImageEnhencer(color_jitter=True, green=False),
        ApplyCLAHE(green=green),
        ToTensor(green=green),
    ]
    transformation = torchvision.transforms.Compose(transforms_list)

    # 验证集仅做必要处理
    transforms_val = [
        Resize(642),
        CenterCrop(640),
        ApplyCLAHE(green=green),
        ToTensor(green=green)
    ]
    transformation_val = torchvision.transforms.Compose(transforms_val)

    print('\t Loading Train and Validation Datasets... \n')
    train_data = IDRiDDataset(mode='train', tasks=tasks, transform=transformation, root_dir=data_path)
    val_data = IDRiDDataset(mode='val', tasks=tasks, transform=transformation_val, root_dir=data_path)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, drop_last=False)

    print('Train dataset size:', len(train_loader.dataset))
    print('Val dataset size:', len(val_loader.dataset))

    return train_loader, val_loader


if __name__ == '__main__':
    # 测试代码，请根据实际路径修改
    path = '/path/to/your/DDR_dataset'
    train_loader, val_loader = load_train_val_data(tasks=['MA'], data_path=path, green=True)