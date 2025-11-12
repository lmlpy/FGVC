import os
import shutil
import random
from pathlib import Path
import argparse

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='数据分割脚本')
    # 数据参数
    parser.add_argument('--raw_dir', type=str, default='./data/raw', help='原始数据目录')
    parser.add_argument('--train_dir', type=str, default='./data/train',help='训练数据目录')
    parser.add_argument('--val_dir', type=str, default='./data/val', help='验证数据目录')
    parser.add_argument('--split_ratio', type=float, default=0.8, help='训练集占比')

    return parser.parse_args()

def split_dataset_with_info(raw_data_dir, train_dir, val_dir, split_ratio=0.8, seed=42):
    # 设置随机种子
    random.seed(seed)

    # 创建目标目录
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # 统计信息
    stats = {
        'total_classes': 0,
        'total_images': 0,
        'train_images': 0,
        'val_images': 0,
        'class_details': {}
    }

    # 获取所有类别文件夹
    class_dirs = sorted([d for d in os.listdir(raw_data_dir) if os.path.isdir(os.path.join(raw_data_dir, d))])

    if not class_dirs:
        print(f"在 {raw_data_dir} 中没有找到任何类别文件夹")
        return

    stats['total_classes'] = len(class_dirs)
    print(f"找到 {len(class_dirs)} 个类别")
    print("=" * 50)

    for class_name in class_dirs:
        class_path = os.path.join(raw_data_dir, class_name)

        # 获取该类别的所有图片文件
        image_files = []
        for ext in ['*.jpg']:
            image_files.extend(Path(class_path).glob(f"**/{ext}"))

        image_files = [str(f) for f in image_files]

        if not image_files:
            print(f"警告: 在 {class_name} 中没有找到图片文件")
            continue

        # 随机打乱并分割
        random.shuffle(image_files)
        split_point = int(len(image_files) * split_ratio)
        train_files = image_files[:split_point]
        val_files = image_files[split_point:]

        # 创建类别子目录并复制文件
        train_class_dir = os.path.join(train_dir, class_name)
        val_class_dir = os.path.join(val_dir, class_name)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(val_class_dir, exist_ok=True)

        for file_path in train_files:
            filename = os.path.basename(file_path)
            shutil.copy2(file_path, os.path.join(train_class_dir, filename))

        for file_path in val_files:
            filename = os.path.basename(file_path)
            shutil.copy2(file_path, os.path.join(val_class_dir, filename))

        # 更新统计信息
        stats['total_images'] += len(image_files)
        stats['train_images'] += len(train_files)
        stats['val_images'] += len(val_files)
        stats['class_details'][class_name] = {
            'total': len(image_files),
            'train': len(train_files),
            'val': len(val_files)
        }

        print(
            f"{class_name:<20} | 总数: {len(image_files):<4} | 训练: {len(train_files):<4} | 验证: {len(val_files):<4}")

    # 打印总结
    print("=" * 50)
    print(f"数据集分割总结:")
    print(f"类别数量: {stats['total_classes']}")
    print(f"图片总数: {stats['total_images']}")
    print(f"训练集: {stats['train_images']} ({stats['train_images'] / stats['total_images'] * 100:.1f}%)")
    print(f"验证集: {stats['val_images']} ({stats['val_images'] / stats['total_images'] * 100:.1f}%)")
    print(f"训练集路径: {train_dir}")
    print(f"验证集路径: {val_dir}")


# 使用示例
if __name__ == "__main__":
    args = parse_args()
    # 路径配置
    raw_data_dir = args.raw_dir  # 原始数据目录
    train_dir = args.train_dir  # 训练集保存目录
    val_dir = args.val_dir  # 验证集保存目录

    # 检查原始数据是否存在
    if not os.path.exists(raw_data_dir):
        print(f"错误: 原始数据目录 {raw_data_dir} 不存在")
        exit(1)

    print("开始分割数据集...")

    # 方法1: 基础版本
    # split_dataset(raw_data_dir, train_dir, val_dir, split_ratio=0.8, seed=42)

    # 方法2: 带详细统计信息的版本（推荐）
    split_dataset_with_info(raw_data_dir, train_dir, val_dir, split_ratio=0.8, seed=42)
