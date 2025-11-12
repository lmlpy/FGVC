import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import pandas as pd
from pathlib import Path
import json
import time


def calculate_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    计算准确率

    Args:
        predictions: 预测标签 [batch_size] or [batch_size, num_classes]
        targets: 真实标签 [batch_size]

    Returns:
        float: 准确率
    """
    if predictions.dim() > 1:
        predictions = torch.argmax(predictions, dim=1)
    correct = (predictions == targets).float().sum()
    accuracy = correct / targets.size(0)
    return accuracy.item()


def calculate_precision_recall_f1(predictions: torch.Tensor, targets: torch.Tensor,
                                  average: str = 'macro') -> Dict[str, float]:
    """
    计算精确率、召回率和F1分数

    Args:
        predictions: 预测标签
        targets: 真实标签
        average: 平均方法 ('micro', 'macro', 'weighted')

    Returns:
        dict: 包含precision, recall, f1的字典
    """
    predictions_np = predictions.cpu().numpy() if isinstance(predictions, torch.Tensor) else predictions
    targets_np = targets.cpu().numpy() if isinstance(targets, torch.Tensor) else targets

    if predictions_np.ndim > 1:
        predictions_np = np.argmax(predictions_np, axis=1)

    precision = precision_score(targets_np, predictions_np, average=average, zero_division=0)
    recall = recall_score(targets_np, predictions_np, average=average, zero_division=0)
    f1 = f1_score(targets_np, predictions_np, average=average, zero_division=0)

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def calculate_confusion_matrix(predictions: torch.Tensor, targets: torch.Tensor,
                               class_names: Optional[List[str]] = None) -> np.ndarray:
    """
    计算混淆矩阵

    Args:
        predictions: 预测标签
        targets: 真实标签
        class_names: 类别名称列表

    Returns:
        np.ndarray: 混淆矩阵
    """
    predictions_np = predictions.cpu().numpy() if isinstance(predictions, torch.Tensor) else predictions
    targets_np = targets.cpu().numpy() if isinstance(targets, torch.Tensor) else targets

    if predictions_np.ndim > 1:
        predictions_np = np.argmax(predictions_np, axis=1)

    if class_names is None:
        num_classes = len(np.unique(targets_np))
        class_names = [f'Class {i}' for i in range(num_classes)]

    cm = confusion_matrix(targets_np, predictions_np, labels=range(len(class_names)))
    return cm


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str],
                          save_path: Optional[str] = None,
                          figsize: Tuple[int, int] = (10, 8)) -> None:
    """
    绘制混淆矩阵热力图

    Args:
        cm: 混淆矩阵
        class_names: 类别名称列表
        save_path: 保存路径
        figsize: 图像尺寸
    """
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def calculate_auc_roc(probabilities: torch.Tensor, targets: torch.Tensor,
                      average: str = 'macro') -> float:
    """
    计算AUC-ROC分数（多分类）

    Args:
        probabilities: 预测概率 [batch_size, num_classes]
        targets: 真实标签 [batch_size]
        average: 平均方法

    Returns:
        float: AUC-ROC分数
    """
    probabilities_np = probabilities.cpu().numpy() if isinstance(probabilities, torch.Tensor) else probabilities
    targets_np = targets.cpu().numpy() if isinstance(targets, torch.Tensor) else targets

    # 对于多分类，使用one-vs-rest策略
    if probabilities_np.shape[1] > 2:
        auc = roc_auc_score(targets_np, probabilities_np, multi_class='ovr', average=average)
    else:
        auc = roc_auc_score(targets_np, probabilities_np[:, 1])

    return auc


def calculate_average_precision(probabilities: torch.Tensor, targets: torch.Tensor,
                                average: str = 'macro') -> float:
    """
    计算平均精确率 (Average Precision)

    Args:
        probabilities: 预测概率
        targets: 真实标签
        average: 平均方法

    Returns:
        float: 平均精确率
    """
    probabilities_np = probabilities.cpu().numpy() if isinstance(probabilities, torch.Tensor) else probabilities
    targets_np = targets.cpu().numpy() if isinstance(targets, torch.Tensor) else targets

    # 将目标标签转换为one-hot编码
    targets_one_hot = np.eye(probabilities_np.shape[1])[targets_np]

    ap = average_precision_score(targets_one_hot, probabilities_np, average=average)
    return ap


def calculate_per_class_metrics(predictions: torch.Tensor, targets: torch.Tensor,
                                class_names: Optional[List[str]] = None) -> pd.DataFrame:
    """
    计算每个类别的评估指标

    Args:
        predictions: 预测标签
        targets: 真实标签
        class_names: 类别名称列表

    Returns:
        pd.DataFrame: 每个类别的指标数据框
    """
    predictions_np = predictions.cpu().numpy() if isinstance(predictions, torch.Tensor) else predictions
    targets_np = targets.cpu().numpy() if isinstance(targets, torch.Tensor) else targets

    if predictions_np.ndim > 1:
        predictions_np = np.argmax(predictions_np, axis=1)

    if class_names is None:
        num_classes = len(np.unique(targets_np))
        class_names = [f'Class {i}' for i in range(num_classes)]

    # 计算每个类别的指标
    per_class_precision = precision_score(targets_np, predictions_np, average=None, zero_division=0)
    per_class_recall = recall_score(targets_np, predictions_np, average=None, zero_division=0)
    per_class_f1 = f1_score(targets_np, predictions_np, average=None, zero_division=0)

    # 创建结果数据框
    results = pd.DataFrame({
        'Class': class_names,
        'Precision': per_class_precision,
        'Recall': per_class_recall,
        'F1-Score': per_class_f1
    })

    return results


def generate_classification_report(predictions: torch.Tensor, targets: torch.Tensor,
                                   class_names: Optional[List[str]] = None) -> str:
    """
    生成详细的分类报告

    Args:
        predictions: 预测标签
        targets: 真实标签
        class_names: 类别名称列表

    Returns:
        str: 分类报告
    """
    predictions_np = predictions.cpu().numpy() if isinstance(predictions, torch.Tensor) else predictions
    targets_np = targets.cpu().numpy() if isinstance(targets, torch.Tensor) else targets

    if predictions_np.ndim > 1:
        predictions_np = np.argmax(predictions_np, axis=1)

    if class_names is None:
        num_classes = len(np.unique(targets_np))
        class_names = [str(i) for i in range(num_classes)]

    report = classification_report(targets_np, predictions_np,
                                   target_names=class_names, digits=4)
    return report


def calculate_top_k_accuracy(outputs: torch.Tensor, targets: torch.Tensor, k: int = 5) -> float:
    """
    计算Top-K准确率

    Args:
        outputs: 模型输出 [batch_size, num_classes]
        targets: 真实标签 [batch_size]
        k: Top-K值

    Returns:
        float: Top-K准确率
    """
    _, pred = outputs.topk(k, dim=1, largest=True, sorted=True)
    correct = pred.eq(targets.view(-1, 1).expand_as(pred))
    correct_k = correct[:, :k].reshape(-1).float().sum(0)
    accuracy = correct_k / targets.size(0)
    return accuracy.item()


def evaluate_model(model: nn.Module, dataloader: torch.utils.data.DataLoader,
                   device: torch.device, num_classes: int) -> Dict[str, float]:
    """
    完整评估模型性能

    Args:
        model: 模型
        dataloader: 数据加载器
        device: 设备
        num_classes: 类别数量

    Returns:
        dict: 包含所有评估指标的字典
    """
    model.eval()
    all_predictions = []
    all_targets = []
    all_probabilities = []

    with torch.no_grad():
        for batch_idx, (data, target, _) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            output = model(data)

            probabilities = torch.softmax(output, dim=1)
            predictions = torch.argmax(output, dim=1)

            all_predictions.append(predictions.cpu())
            all_targets.append(target.cpu())
            all_probabilities.append(probabilities.cpu())

    # 合并所有批次的结果
    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)
    all_probabilities = torch.cat(all_probabilities)

    # 计算各种指标
    accuracy = calculate_accuracy(all_predictions, all_targets)
    prf_metrics = calculate_precision_recall_f1(all_predictions, all_targets)
    auc_roc = calculate_auc_roc(all_probabilities, all_targets)
    avg_precision = calculate_average_precision(all_probabilities, all_targets)

    # 计算Top-3和Top-5准确率（如果类别数足够）
    top3_acc = 0.0
    top5_acc = 0.0
    if num_classes >= 3:
        outputs_for_topk = torch.cat([batch[0] for batch in all_probabilities])
        top3_acc = calculate_top_k_accuracy(outputs_for_topk, all_targets, k=3)
    if num_classes >= 5:
        top5_acc = calculate_top_k_accuracy(outputs_for_topk, all_targets, k=5)

    results = {
        'accuracy': accuracy,
        'precision': prf_metrics['precision'],
        'recall': prf_metrics['recall'],
        'f1_score': prf_metrics['f1'],
        'auc_roc': auc_roc,
        'average_precision': avg_precision,
        'top3_accuracy': top3_acc,
        'top5_accuracy': top5_acc
    }

    return results


def save_metrics_to_file(metrics: Dict, file_path: str) -> None:
    """
    将评估指标保存到文件

    Args:
        metrics: 评估指标字典
        file_path: 文件路径
    """
    # 确保目录存在
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)

    # 转换numpy类型为Python原生类型
    metrics_serializable = {}
    for key, value in metrics.items():
        if isinstance(value, (np.float32, np.float64)):
            metrics_serializable[key] = float(value)
        elif isinstance(value, np.ndarray):
            metrics_serializable[key] = value.tolist()
        else:
            metrics_serializable[key] = value

    with open(file_path, 'w') as f:
        json.dump(metrics_serializable, f, indent=4)


def print_metrics(metrics: Dict[str, float], title: str = "评估结果") -> None:
    """
    美观地打印评估指标

    Args:
        metrics: 评估指标字典
        title: 标题
    """
    print(f"\n{'=' * 50}")
    print(f"{title:^50}")
    print(f"{'=' * 50}")

    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key.replace('_', ' ').title():<20}: {value:.4f}")
        else:
            print(f"{key.replace('_', ' ').title():<20}: {value}")

    print(f"{'=' * 50}")


# 测试函数
def test_metrics_functions():
    """测试所有评估指标函数"""
    print("测试评估指标函数...")

    # 生成模拟数据
    batch_size = 100
    num_classes = 5

    # 模拟预测和真实标签
    torch.manual_seed(42)
    predictions = torch.randint(0, num_classes, (batch_size,))
    targets = torch.randint(0, num_classes, (batch_size,))
    probabilities = torch.softmax(torch.randn(batch_size, num_classes), dim=1)

    # 测试各个函数
    accuracy = calculate_accuracy(predictions, targets)
    print(f"准确率: {accuracy:.4f}")

    prf_metrics = calculate_precision_recall_f1(predictions, targets)
    print(f"精确率: {prf_metrics['precision']:.4f}")
    print(f"召回率: {prf_metrics['recall']:.4f}")
    print(f"F1分数: {prf_metrics['f1']:.4f}")

    cm = calculate_confusion_matrix(predictions, targets)
    print("混淆矩阵:")
    print(cm)

    auc_roc = calculate_auc_roc(probabilities, targets)
    print(f"AUC-ROC: {auc_roc:.4f}")

    per_class_df = calculate_per_class_metrics(predictions, targets)
    print("\n每个类别指标:")
    print(per_class_df)

    # 测试完整评估
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, num_classes)

        def forward(self, x):
            return self.fc(x)

    # 模拟数据加载器
    class MockDataset:
        def __len__(self):
            return 100

        def __getitem__(self, idx):
            return torch.randn(10), torch.randint(0, num_classes, (1,)), f"sample_{idx}"

    mock_loader = torch.utils.data.DataLoader(MockDataset(), batch_size=10)
    model = SimpleModel()
    device = torch.device('cpu')

    metrics = evaluate_model(model, mock_loader, device, num_classes)
    print_metrics(metrics, "完整模型评估测试")


if __name__ == "__main__":
    test_metrics_functions()