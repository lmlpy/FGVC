import argparse
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from models import SimpleNet
from datasets import create_data_loaders
from utils import calculate_accuracy, evaluate_model, print_metrics
import json
from pathlib import Path
import numpy as np
from torch.utils.data import Subset


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='图像分类训练脚本')
    # 数据参数
    parser.add_argument('--train_dir', type=str, default='../data/train',help='训练数据目录')
    parser.add_argument('--val_dir', type=str, default='../data/val', help='验证数据目录')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载工作进程数')

    # 模型参数
    parser.add_argument('--model_name', type=str, default='SimpleNet', choices=['SimpleNet', 'SimpleNetV2'],help='模型名称')
    parser.add_argument('--num_classes', type=int, default=10, help='类别数量')
    parser.add_argument('--input_size', type=int, nargs=2, default=[224, 224], help='输入图像尺寸 [height, width]')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减')
    parser.add_argument('--momentum', type=float, default=0.9, help='动量')

    # 评估参数
    parser.add_argument('--val_steps', type=int, default=100, help='每隔多少步进行验证')
    parser.add_argument('--save_steps', type=int, default=500, help='每隔多少步保存模型')

    # 保存路径
    parser.add_argument('--output_dir', type=str, default='../log/outputs', help='输出目录')
    parser.add_argument('--experiment_name', type=str, default='exp1', help='实验名称')

    # 其他参数
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的检查点路径')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu', 'cuda:0', 'cuda:1'], help='设备类型')

    return parser.parse_args()

def setup_environment(args):
    """设置训练环境"""
    # 设置设备
    if args.device != 'cpu' and torch.cuda.is_available():
        device = torch.device(args.device)
        print(f"使用 GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        print("使用 CPU")

    # 创建输出目录
    experiment_dir = Path(args.output_dir) / args.experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # 保存训练参数
    with open(experiment_dir / 'args.json', 'w') as f:
        json.dump(vars(args), f, indent=4)

    return device, experiment_dir

def create_model_and_optimizer(args, device, num_classes):
    """创建模型和优化器"""
    # 创建模型
    model = SimpleNet(
        num_classes=num_classes,
        device=device
    )
    model.to(device)

    # 创建优化器
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # 创建学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs
    )

    # 损失函数
    criterion = nn.CrossEntropyLoss()

    return model, optimizer, scheduler, criterion

def save_class_mapping(class_to_idx, save_path):
    mapping_data = {
        'class_to_idx': class_to_idx,
        'idx_to_class': {v: k for k, v in class_to_idx.items()},
        'num_classes': len(class_to_idx),
        'save_time': time.strftime('%Y-%m-%d %H:%M:%S')
    }

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(mapping_data, f, indent=4, ensure_ascii=False)

    print(f"类别映射已保存: {save_path}")

def create_compatible_sampled_val_loader(val_dataset, sample_ratio=0.1, batch_size=32, seed=42):
    """
    创建兼容的采样验证集加载器
    """
    import numpy as np
    from torch.utils.data import DataLoader

    # 设置随机种子
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 计算采样数量
    total_size = len(val_dataset)
    sample_size = int(total_size * sample_ratio)

    # 生成随机索引
    indices = np.random.choice(total_size, sample_size, replace=False)

    # 创建自定义采样数据集类
    class SampledDataset(torch.utils.data.Dataset):
        def __init__(self, original_dataset, indices):
            self.original_dataset = original_dataset
            self.indices = indices

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, idx):
            # 获取原始数据
            original_data = self.original_dataset[self.indices[idx]]

            # 确保返回格式一致 (image, label, path)
            if len(original_data) == 3:
                return original_data  # (image, label, path)
            elif len(original_data) == 2:
                image, label = original_data
                # 创建一个虚拟路径
                path = f"sampled_{idx}"
                return image, label, path
            else:
                raise ValueError(f"未知的数据格式: {len(original_data)} 个元素")

    # 创建采样数据集
    sampled_dataset = SampledDataset(val_dataset, indices)

    # 创建数据加载器
    sampled_loader = DataLoader(
        sampled_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    print(f"验证集采样: {total_size} -> {sample_size} 样本")
    print(f"采样比例: {sample_ratio:.1%}")

    return sampled_loader

def train_epoch(model, train_loader, val_loader, optimizer, criterion, device, epoch, args, global_step):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    total_samples = 0

    start_time = time.time()

    for batch_idx, (images, labels, _) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 统计信息
        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        acc = calculate_accuracy(outputs, labels)
        running_acc += acc * batch_size
        total_samples += batch_size

        global_step += 1

        # 打印训练信息
        if batch_idx % 10 == 0:
            avg_loss = running_loss / total_samples
            avg_acc = running_acc / total_samples
            print(f'Epoch: {epoch} [{batch_idx}/{len(train_loader)}] '
                  f'Loss: {avg_loss:.4f} Acc: {avg_acc:.4f} '
                  f'LR: {optimizer.param_groups[0]["lr"]:.6f}')

        # 验证步骤
        if args.val_steps != 0 and global_step % args.val_steps == 0:
            val_metrics = validate_model(model, val_loader, criterion, device, args.num_classes)
            print_metrics(val_metrics, f"验证结果 - Step {global_step}")

            # 切换回训练模式
            model.train()

    epoch_loss = running_loss / total_samples
    epoch_acc = running_acc / total_samples
    epoch_time = time.time() - start_time

    return epoch_loss, epoch_acc, epoch_time, global_step

def validate_model(model, val_loader, criterion, device, num_classes):
    """验证模型"""
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    total_samples = 0

    with torch.no_grad():
        for images, labels, _ in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            acc = calculate_accuracy(outputs, labels)
            running_acc += acc * batch_size
            total_samples += batch_size

    val_loss = running_loss / total_samples
    val_acc = running_acc / total_samples

    return {
        'val_loss': val_loss,
        'val_accuracy': val_acc
    }

def save_checkpoint(model, optimizer, scheduler, epoch, global_step, metrics, save_path):
    """保存检查点"""
    checkpoint = {
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'metrics': metrics
    }
    torch.save(checkpoint, save_path)
    print(f"检查点已保存: {save_path}")

def load_checkpoint(checkpoint_path, model, optimizer, scheduler, device):
    """加载检查点"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch']
    global_step = checkpoint['global_step']
    metrics = checkpoint.get('metrics', {})

    print(f"从检查点恢复训练: {checkpoint_path}")
    print(f"恢复的轮数: {epoch}, 全局步数: {global_step}")

    return epoch, global_step, metrics

def main():
    """主训练函数"""
    # 解析参数
    args = parse_args()
    print("训练参数:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")

    # 设置环境
    device, experiment_dir = setup_environment(args)

    # 创建数据加载器
    print("创建数据加载器...")
    train_loader, val_loader, train_dataset, val_dataset = create_data_loaders(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        batch_size=args.batch_size,
        target_size=tuple(args.input_size),
        num_workers=args.num_workers
    )
    #val_loader = create_compatible_sampled_val_loader(val_loader, sample_ratio=0.05, batch_size=args.batch_size)

    # 保存类别映射
    class_to_idx = train_dataset.class_to_idx
    mapping_path = experiment_dir / 'class_mapping.json'
    save_class_mapping(class_to_idx, mapping_path)

    # 获取实际类别数
    actual_num_classes = len(train_dataset.class_to_idx)
    if args.num_classes != actual_num_classes:
        print(f"更新类别数: {args.num_classes} -> {actual_num_classes}")
        args.num_classes = actual_num_classes

    # 创建模型和优化器
    print("创建模型和优化器...")
    model, optimizer, scheduler, criterion = create_model_and_optimizer(
        args, device, args.num_classes
    )

    # 恢复训练
    start_epoch = 0
    global_step = 0
    best_val_acc = 0.0
    train_history = []

    if args.resume:
        start_epoch, global_step, _ = load_checkpoint(
            args.resume, model, optimizer, scheduler, device
        )
        start_epoch += 1  # 从下一轮开始

    print(f"开始训练...")
    print(f"训练集: {len(train_dataset)} 张图片")
    print(f"验证集: {len(val_dataset)} 张图片")
    print(f"类别数: {args.num_classes}")
    print(f"模型: {args.model_name}")

    # 训练循环
    for epoch in range(start_epoch, args.epochs):
        print(f"\n{'=' * 60}")
        print(f'Epoch {epoch + 1}/{args.epochs}')
        print(f"{'=' * 60}")

        # 训练一个epoch
        train_loss, train_acc, epoch_time, global_step = train_epoch(
            model, train_loader, val_loader, optimizer, criterion, device,
            epoch + 1, args, global_step
        )

        # 验证
        val_metrics = validate_model(model, val_loader, criterion, device, args.num_classes)

        # 更新学习率
        scheduler.step()

        # 记录历史
        epoch_metrics = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_accuracy': train_acc,
            'val_loss': val_metrics['val_loss'],
            'val_accuracy': val_metrics['val_accuracy'],
            'learning_rate': optimizer.param_groups[0]['lr'],
            'epoch_time': epoch_time
        }
        train_history.append(epoch_metrics)

        # 打印epoch结果
        print(f"\nEpoch {epoch + 1} 结果:")
        print(f"  训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.4f}")
        print(f"  验证损失: {val_metrics['val_loss']:.4f}, 验证准确率: {val_metrics['val_accuracy']:.4f}")
        print(f"  学习率: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"  耗时: {epoch_time:.2f}秒")

        # 保存最佳模型
        if val_metrics['val_accuracy'] > best_val_acc:
            best_val_acc = val_metrics['val_accuracy']
            save_checkpoint(
                model, optimizer, scheduler, epoch, global_step,
                epoch_metrics, experiment_dir / 'best_model.pth'
            )
            print(f"新的最佳模型已保存! 验证准确率: {best_val_acc:.4f}")

        # 定期保存检查点
        if (epoch + 1) % 5 == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, global_step,
                epoch_metrics, experiment_dir / f'checkpoint_epoch_{epoch + 1}.pth'
            )

    # 保存最终模型
    save_checkpoint(
        model, optimizer, scheduler, args.epochs - 1, global_step,
        train_history[-1], experiment_dir / 'final_model.pth'
    )

    # 保存训练历史
    with open(experiment_dir / 'training_history.json', 'w') as f:
        json.dump(train_history, f, indent=4)

    print(f"\n训练完成!")
    print(f"最佳验证准确率: {best_val_acc:.4f}")
    print(f"所有结果保存在: {experiment_dir}")


if __name__ == "__main__":
    main()