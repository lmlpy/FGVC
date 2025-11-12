import argparse
import os
import glob
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import pandas as pd
from pathlib import Path
import json
from models import SimpleNet
from datasets import ClassificationDataset
import numpy as np


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='图像分类预测脚本')

    parser.add_argument('--test_dir', type=str, default='test',
                        help='测试图片目录')
    parser.add_argument('--model_path', type=str, required=True,
                        help='训练好的模型路径')
    parser.add_argument('--class_mapping', type=str, required=True,
                        help='类别映射文件路径')
    parser.add_argument('--output_csv', type=str, default='predictions.csv',
                        help='输出CSV文件路径')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='预测批次大小')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cuda:0', 'cuda:1', 'cpu'],
                        help='设备类型')

    return parser.parse_args()


def load_class_mapping(mapping_path):
    """
    加载类别映射文件

    Args:
        mapping_path: 类别映射文件路径

    Returns:
        tuple: (class_to_idx, idx_to_class)
    """
    try:
        with open(mapping_path, 'r', encoding='utf-8') as f:
            mapping_data = json.load(f)

        if 'class_to_idx' in mapping_data:
            class_to_idx = mapping_data['class_to_idx']
            idx_to_class = {int(v): k for k, v in class_to_idx.items()}
        else:
            # 如果文件直接就是映射
            class_to_idx = mapping_data
            idx_to_class = {int(v): k for k, v in mapping_data.items()}

        print(f"类别映射加载成功: {mapping_path}")
        print(f"类别数量: {len(class_to_idx)}")
        print(f"类别映射: {class_to_idx}")

        return class_to_idx, idx_to_class

    except Exception as e:
        print(f"错误: 无法加载类别映射文件 {mapping_path}: {e}")
        return None, None

def load_model(model_path, num_classes, device):
    """
    加载模型

    Args:
        model_path: 模型路径
        num_classes: 类别数量
        device: 设备

    Returns:
        model: 加载的模型
    """
    # 加载检查点
    checkpoint = torch.load(model_path, map_location=device)

    # 获取模型参数
    if 'model_state_dict' in checkpoint:
        model_state_dict = checkpoint['model_state_dict']
    else:
        model_state_dict = checkpoint

    # 创建模型
    model = SimpleNet(num_classes=num_classes, device=device)
    model.load_state_dict(model_state_dict)
    model.to(device)
    model.eval()

    print(f"模型加载成功: {model_path}")
    return model

def format_class_name(class_name, padding=4):
    """
    格式化类别名，确保是4位数字，不足补0

    Args:
        class_name: 原始类别名
        padding: 补齐位数

    Returns:
        str: 格式化后的类别名
    """
    # 如果类别名已经是数字，直接格式化
    if class_name.isdigit():
        return class_name.zfill(padding)

    # 如果类别名包含数字，提取数字部分
    import re
    numbers = re.findall(r'\d+', class_name)
    if numbers:
        return numbers[0].zfill(padding)

    # 如果没有数字，返回原始名称（这种情况应该避免）
    print(f"警告: 类别名 '{class_name}' 不包含数字，无法格式化")
    return class_name

class TestDataset(torch.utils.data.Dataset):
    """测试数据集类"""

    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"错误: 无法加载图像 {img_path}: {e}")
            # 创建黑色图像作为替代
            image = Image.new('RGB', (224, 224), color='black')

        if self.transform:
            image = self.transform(image)

        filename = Path(img_path).name
        return image, filename, img_path

def predict_images(model, test_dir, transform, device, batch_size=32):
    """
    预测所有测试图片

    Args:
        model: 训练好的模型
        test_dir: 测试图片目录
        transform: 图像变换
        device: 设备
        batch_size: 批次大小

    Returns:
        list: 预测结果列表
    """
    # 获取所有测试图片
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_paths = []

    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(test_dir, ext)))
        image_paths.extend(glob.glob(os.path.join(test_dir, ext.upper())))

    if not image_paths:
        raise ValueError(f"在 {test_dir} 中没有找到任何测试图片")

    print(f"找到 {len(image_paths)} 张测试图片")

    # 创建测试数据集和数据加载器
    test_dataset = TestDataset(image_paths, transform=transform)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    results = []

    with torch.no_grad():
        for batch_idx, (images, filenames, paths) in enumerate(test_loader):
            images = images.to(device)

            # 预测
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)
            confidences = torch.max(probabilities, dim=1)[0]

            # 处理批次结果
            for i in range(len(filenames)):
                pred_class_idx = predictions[i].item()
                confidence = confidences[i].item()

                results.append({
                    'filename': filenames[i],
                    'predicted_class_idx': pred_class_idx,
                    'confidence': confidence,
                    'filepath': paths[i]
                })

            if (batch_idx + 1) % 10 == 0:
                print(f"已处理 {batch_idx + 1}/{len(test_loader)} 批次")

    return results


def save_predictions(results, idx_to_class, output_csv):
    """
    保存预测结果到CSV文件

    Args:
        results: 预测结果列表
        idx_to_class: 索引到类别名的映射
        output_csv: 输出CSV文件路径
    """
    # 准备数据
    data = []
    for result in results:
        filename = result['filename']
        pred_class_idx = result['predicted_class_idx']

        # 获取类别名
        if idx_to_class and pred_class_idx in idx_to_class:
            class_name = idx_to_class[pred_class_idx]
            formatted_class = format_class_name(class_name)
        else:
            # 如果没有类别映射，使用索引
            formatted_class = str(pred_class_idx).zfill(4)
            print(f"警告: 类别索引 {pred_class_idx} 不在映射中，使用默认映射")

        data.append([filename, formatted_class])

    # 创建DataFrame并保存
    df = pd.DataFrame(data, columns=['filename', 'class'])

    # 按文件名排序
    df = df.sort_values('filename').reset_index(drop=True)

    # 保存CSV
    output_dirname = os.path.dirname(output_csv)
    os.makedirs(output_dirname, exist_ok=True)
    df.to_csv(output_csv, header=False, index=False, encoding='utf-8')

    print(f"预测结果已保存到: {output_csv}")
    print(f"总共预测了 {len(data)} 张图片")

    # 显示前几个结果
    print("\n前5个预测结果:")
    print(df.head())

    return df


def main():
    """主预测函数"""
    args = parse_args()

    print("预测参数:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")

    # 设置设备
    if args.device != 'cpu' and torch.cuda.is_available():
        device = torch.device(args.device)
        print(f"使用 GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        print("使用 CPU")

    # 加载类别映射
    class_to_idx, idx_to_class = load_class_mapping(args.class_mapping)
    if idx_to_class is None:
        print("错误: 无法加载类别映射，退出预测")
        return

    num_classes = len(class_to_idx)

    # 加载模型
    model = load_model(args.model_path, num_classes, device)

    # 创建预处理变换（与训练时验证集保持一致）
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 执行预测
    results = predict_images(
        model, args.test_dir, transform, device, args.batch_size
    )

    # 保存结果
    df = save_predictions(results, idx_to_class, args.output_csv)

    # 统计预测分布
    class_distribution = df['class'].value_counts().sort_index()
    print(f"\n预测类别分布:")
    for class_name, count in class_distribution.items():
        print(f"  {class_name}: {count} 张图片")


if __name__ == "__main__":
    main()