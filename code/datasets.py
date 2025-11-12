import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
import random


class ClassificationDataset(Dataset):
    """
    分类任务数据集类
    """

    def __init__(self, data_dir, mode='train', transform=None, target_size=(224, 224),
                 augmentation=True, seed=42):
        self.data_dir = data_dir
        self.mode = mode
        self.transform = transform
        self.target_size = target_size
        self.augmentation = augmentation and mode == 'train'
        self.seed = seed

        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}
        self.idx_to_class = {}

        # 设置随机种子
        random.seed(seed)
        torch.manual_seed(seed)

        self._load_data()
        self._setup_transforms()

        print(f"初始化 {mode} 数据集完成")
        print(f"数据目录: {data_dir}")
        print(f"类别数量: {len(self.class_to_idx)}")
        print(f"图片数量: {len(self.image_paths)}")
        print(f"类别映射: {self.class_to_idx}")

    def _load_data(self):
        """加载数据路径和标签"""
        # 检查数据目录是否存在
        if not os.path.exists(self.data_dir):
            raise ValueError(f"数据目录不存在: {self.data_dir}")

        # 获取所有类别
        classes = sorted([d for d in os.listdir(self.data_dir)
                          if os.path.isdir(os.path.join(self.data_dir, d))])

        if not classes:
            raise ValueError(f"在 {self.data_dir} 中没有找到任何类别文件夹")

        # 创建类别映射
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
        self.idx_to_class = {idx: cls_name for idx, cls_name in enumerate(classes)}

        # 收集所有图像路径和标签
        for class_name in classes:
            class_dir = os.path.join(self.data_dir, class_name)
            class_idx = self.class_to_idx[class_name]

            # 支持多种图像格式
            image_extensions = ['*.jpg', '*.png']
            for ext in image_extensions:
                image_paths = list(Path(class_dir).glob(f"**/{ext}"))
                for img_path in image_paths:
                    self.image_paths.append(str(img_path))
                    self.labels.append(class_idx)

        # 打乱训练数据
        if self.mode == 'train':
            combined = list(zip(self.image_paths, self.labels))
            random.shuffle(combined)
            self.image_paths, self.labels = zip(*combined)
            self.image_paths = list(self.image_paths)
            self.labels = list(self.labels)

    def _setup_transforms(self):
        """设置图像变换"""
        if self.transform is not None:
            return

        if self.mode == 'train' and self.augmentation:
            # 训练模式：数据增强
            self.transform = transforms.Compose([
                transforms.Resize(self.target_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomResizedCrop(self.target_size, scale=(0.8, 1.0)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            # 验证/测试模式：仅基础变换
            self.transform = transforms.Compose([
                transforms.Resize(self.target_size),
                transforms.CenterCrop(self.target_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        """返回数据集大小"""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        获取数据项

        Args:
            idx (int): 索引

        Returns:
            tuple: (image, label, image_path)
        """
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # 加载图像
        try:
            # 尝试打开图像
            with open(img_path, 'rb') as f:
                image = Image.open(f)
                # 处理各种图像模式
                if image.mode in ('P', 'PA'):
                    # 调色板图像：转换为RGBA再转RGB
                    image = image.convert('RGBA')
                    background = Image.new('RGBA', image.size, (255, 255, 255, 255))
                    image = Image.alpha_composite(background, image)
                    image = image.convert('RGB')
                elif image.mode in ('RGBA', 'LA'):
                    # 带透明度的图像：合并到白色背景
                    background = Image.new('RGBA', image.size, (255, 255, 255, 255))
                    image = Image.alpha_composite(background, image)
                    image = image.convert('RGB')
                elif image.mode != 'RGB':
                    # 其他模式直接转换
                    image = image.convert('RGB')

                # 确保图像完全加载
                image.load()
        except Exception as e:
            print(f"警告: 无法加载图像 {img_path}: {e}")
            # 返回一个随机图像作为替代
            image = Image.new('RGB', self.target_size, color=(
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255)
            ))
            print(label)

        # 应用变换
        if self.transform:
            image = self.transform(image)

        return image, label, img_path

    def get_class_distribution(self):
        """获取类别分布"""
        class_counts = {}
        for label in self.labels:
            class_name = self.idx_to_class[label]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        return class_counts

    def get_class_weights(self):
        """计算类别权重，用于处理不平衡数据"""
        class_counts = self.get_class_distribution()
        total_samples = len(self.labels)
        weights = []

        for class_idx in range(len(self.class_to_idx)):
            class_name = self.idx_to_class[class_idx]
            count = class_counts.get(class_name, 0)
            weight = total_samples / (len(self.class_to_idx) * count) if count > 0 else 1.0
            weights.append(weight)

        return torch.tensor(weights, dtype=torch.float32)

    def show_sample(self, idx=0):
        """显示样本信息"""
        image, label, path = self[idx]
        print(f"样本 {idx}:")
        print(f"  路径: {path}")
        print(f"  标签: {label} ({self.idx_to_class[label]})")
        print(f"  图像形状: {image.shape}")
        print(f"  图像范围: [{image.min():.3f}, {image.max():.3f}]")
        return image, label, path


def create_data_loaders(train_dir, val_dir, batch_size=32, target_size=(224, 224),
                        num_workers=4, augmentation=True, pin_memory=True):
    """
    创建训练和验证数据加载器

    Args:
        train_dir (str): 训练数据目录
        val_dir (str): 验证数据目录
        batch_size (int): 批次大小
        target_size (tuple): 图像目标尺寸
        num_workers (int): 数据加载工作进程数
        augmentation (bool): 是否使用数据增强
        pin_memory (bool): 是否锁页内存

    Returns:
        tuple: (train_loader, val_loader, train_dataset, val_dataset)
    """

    # 创建数据集
    train_dataset = ClassificationDataset(
        data_dir=train_dir,
        mode='train',
        target_size=target_size,
        augmentation=augmentation
    )

    val_dataset = ClassificationDataset(
        data_dir=val_dir,
        mode='val',
        target_size=target_size,
        augmentation=False  # 验证集不使用数据增强
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # 丢弃最后一个不完整的batch
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # 验证集不需要打乱
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )

    print(f"创建数据加载器完成:")
    print(f"  训练集: {len(train_dataset)} 张图片")
    print(f"  验证集: {len(val_dataset)} 张图片")
    print(f"  批次大小: {batch_size}")
    print(f"  类别数量: {len(train_dataset.class_to_idx)}")

    return train_loader, val_loader, train_dataset, val_dataset


def get_class_names(data_dir):
    """
    获取数据集的类别名称

    Args:
        data_dir (str): 数据目录

    Returns:
        list: 类别名称列表
    """
    classes = sorted([d for d in os.listdir(data_dir)
                      if os.path.isdir(os.path.join(data_dir, d))])
    return classes


# 测试代码
if __name__ == "__main__":
    # 测试数据集
    train_dir = "../data/train"
    val_dir = "../data/val"

    if os.path.exists(train_dir) and os.path.exists(val_dir):
        print("测试数据集加载...")

        # 创建数据加载器
        train_loader, val_loader, train_dataset, val_dataset = create_data_loaders(
            train_dir=train_dir,
            val_dir=val_dir,
            batch_size=4,
            target_size=(224, 224),
            num_workers=0  # 在测试时设为0避免多进程问题
        )

        # 显示训练集类别分布
        print("\n训练集类别分布:")
        train_dist = train_dataset.get_class_distribution()
        for class_name, count in train_dist.items():
            print(f"  {class_name}: {count}")

        # 显示验证集类别分布
        print("\n验证集类别分布:")
        val_dist = val_dataset.get_class_distribution()
        for class_name, count in val_dist.items():
            print(f"  {class_name}: {count}")

        # 测试一个batch
        print("\n测试一个batch:")
        for images, labels, paths in train_loader:
            print(f"Batch图像形状: {images.shape}")
            print(f"Batch标签形状: {labels.shape}")
            print(f"标签: {labels.tolist()}")
            break

    else:
        print(f"测试数据目录不存在: {train_dir} 或 {val_dir}")