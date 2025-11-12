import os.path
import time
import torch.nn.functional as F
import torch
from modelscope import AutoImageProcessor, AutoModel
from transformers.image_utils import load_image

class DINOv3MultiLayerFeatureExtractor(torch.nn.Module):
    def __init__(self, model=None, layer_names=[-1], device='cuda'):
        super().__init__()
        self.model = model
        self.device = device
        self.layer_names = layer_names
        self.features = {}
    def load_dinov3(self):
        pretrained_model_name = "facebook/dinov3-vitl16-pretrain-lvd1689m"
        cache_dir = '/home/yf/.cache/modelscope/hub/models' # 指定本地缓存路径
        model_dir = os.path.join(cache_dir, pretrained_model_name)
        try:
            self.processor = AutoImageProcessor.from_pretrained(model_dir, cache_dir=cache_dir)
            self.model = AutoModel.from_pretrained(
                model_dir,
                cache_dir=cache_dir,
                output_hidden_states=True,  # 启用多层输出
                device_map=self.device,
            )
        except:
            self.processor = AutoImageProcessor.from_pretrained(pretrained_model_name)
            self.model = AutoModel.from_pretrained(
                pretrained_model_name,
                output_hidden_states=True,  # 启用多层输出
                device_map=self.device,
            )
    def hidden_state_to_feature_map(self, hidden_state):
        batch_size, num_tokens, hidden_dim = hidden_state.shape
        patch_size = int((num_tokens - 5) ** 0.5)  # 排除CLS token
        patch_tokens = hidden_state[:, 5:, :]  # 移除CLS token
        feature_map = patch_tokens.reshape(batch_size, patch_size, patch_size, hidden_dim)
        feature_map = feature_map.permute(0, 3, 1, 2)  # [batch, channels, height, width]
        cls_tokens = hidden_state[:, 0, :][:, :, None, None].expand(-1, -1, feature_map.shape[-2], feature_map.shape[-1])
        feature_map = torch.concat([feature_map, cls_tokens], dim=1)
        return feature_map

    def forward(self, x):
        with torch.inference_mode():
            outputs = self.model(x)
        all_hidden_states = outputs.hidden_states  # 长度为 num_layers + 1（包含embedding层）
        cls_token = all_hidden_states[-1][:, 0, :]

        return cls_token


import os
import argparse
import torch
import torch.nn.functional as F
import shutil
from pathlib import Path
import numpy as np
from tqdm import tqdm
from PIL import Image
import glob


class DINOv3DataRefiner:
    def __init__(self, model_path=None, layer_names=[-1], device='cuda'):
        self.device = device #if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.layer_names = layer_names

        # 初始化特征提取器
        self.feature_extractor = DINOv3MultiLayerFeatureExtractor(device=self.device)
        self.feature_extractor.load_dinov3()

        print(f"模型加载完成，使用设备: {self.device}")

    def extract_features_from_folder(self, image_folder, batch_size=8):
        """从文件夹中提取所有图像的特征"""
        # 获取所有图像文件
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(Path(image_folder).glob(f"**/{ext}"))
            image_paths.extend(Path(image_folder).glob(f"**/{ext.upper()}"))

        image_paths = [str(p) for p in image_paths]

        if not image_paths:
            print(f"在 {image_folder} 中没有找到图像文件")
            return [], []

        print(f"找到 {len(image_paths)} 张图像，开始提取特征...")

        features = []
        successful_paths = []

        with torch.no_grad():
            for i in tqdm(range(0, len(image_paths), batch_size), desc="提取特征"):
                batch_paths = image_paths[i:i + batch_size]
                batch_images = []
                batch_successful = []

                # 加载和预处理图像
                for img_path in batch_paths:
                    try:
                        image = Image.open(img_path).convert('RGB')
                        batch_images.append(image)
                        batch_successful.append(img_path)
                    except Exception as e:
                        print(f"无法加载图像 {img_path}: {e}")
                        continue

                if not batch_images:
                    continue

                # 处理图像
                inputs = self.feature_extractor.processor(images=batch_images, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # 提取特征
                cls_tokens = self.feature_extractor(inputs['pixel_values'])

                features.extend(cls_tokens.cpu().numpy())
                successful_paths.extend(batch_successful)

        return np.array(features), successful_paths

    def compute_distances(self, features):
        """计算特征距离"""
        # 计算特征中心
        center = np.mean(features, axis=0)

        # 计算每个样本到中心的余弦距离
        distances = []
        for feature in features:
            cos_sim = F.cosine_similarity(
                torch.tensor(feature).unsqueeze(0),
                torch.tensor(center).unsqueeze(0)
            ).item()
            distance = 1 - cos_sim
            distances.append(distance)

        distances = np.array(distances)

        # 计算统计信息
        stats = {
            'mean': np.mean(distances),
            'std': np.std(distances),
            'min': np.min(distances),
            'max': np.max(distances),
            'median': np.median(distances)
        }

        return center, distances, stats

    def select_samples_by_mode(self, distances, mode, n=None, threshold=None):
        """根据模式选择样本"""
        if mode == 'nearest':  # 保留最近的n个样本
            if n is None:
                raise ValueError("模式1需要指定n参数")
            selected_indices = np.argsort(distances)[:n]
            print(f"模式1: 保留距离最近的 {n} 个样本")

        elif mode == 'remove_farthest':  # 去除最远的n个样本
            if n is None:
                raise ValueError("模式2需要指定n参数")
            selected_indices = np.argsort(distances)[:-n]
            print(f"模式2: 去除距离最远的 {n} 个样本，保留 {len(selected_indices)} 个样本")

        elif mode == 'threshold':  # 保留距离小于阈值的样本
            if threshold is None:
                raise ValueError("模式3需要指定threshold参数")
            selected_indices = np.where(distances < threshold)[0]
            print(f"模式3: 保留距离小于 {threshold} 的样本，共 {len(selected_indices)} 个")

        else:
            raise ValueError(f"未知模式: {mode}")

        return selected_indices

    def refine_single_class(self, raw_path, refined_path, discard_path, mode, n=None, threshold=None, batch_size=8):
        """精炼单个类别的数据"""
        # 创建输出目录
        os.makedirs(refined_path, exist_ok=True)
        os.makedirs(discard_path, exist_ok=True)

        # 提取特征
        features, image_paths = self.extract_features_from_folder(raw_path, batch_size)

        if len(features) == 0:
            print(f"在 {raw_path} 中没有成功提取到特征")
            return 0, 0

        # 计算距离
        center, distances, stats = self.compute_distances(features)

        # 打印统计信息
        print(f"距离统计 - 均值: {stats['mean']:.4f}, 标准差: {stats['std']:.4f}, "
              f"最小值: {stats['min']:.4f}, 最大值: {stats['max']:.4f}")

        # 根据模式选择样本
        selected_indices = self.select_samples_by_mode(distances, mode, n, threshold)

        # 复制选中的样本到精炼目录，其余到丢弃目录
        refined_count = 0
        discard_count = 0

        for i, img_path in enumerate(image_paths):
            src_path = Path(img_path)
            if i in selected_indices:
                dst_path = Path(refined_path) / src_path.name
                shutil.copy2(src_path, dst_path)
                refined_count += 1
            else:
                dst_path = Path(discard_path) / src_path.name
                shutil.copy2(src_path, dst_path)
                discard_count += 1

        print(f"精炼完成: 保留 {refined_count} 张，丢弃 {discard_count} 张")
        return refined_count, discard_count

    def refine_data(self, raw_path, refined_base_path, discard_base_path, mode, n=None, threshold=None, batch_size=8):
        """主精炼函数，支持单类和多类处理"""
        raw_path = Path(raw_path)
        refined_base_path = Path(refined_base_path)
        discard_base_path = Path(discard_base_path)

        # 判断是单类还是多类数据
        if raw_path.is_file():
            print("错误: raw_path 应该是目录而不是文件")
            return

        all_items = list(raw_path.iterdir())
        image_files = [f for f in all_items if
                       f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']]
        subdirs = [d for d in all_items if d.is_dir()]

        if image_files and not subdirs:
            # 单类数据：直接是图片文件
            print("检测到单类数据模式")
            class_name = raw_path.name
            refined_path = refined_base_path / class_name
            discard_path = discard_base_path / class_name

            refined_count, discard_count = self.refine_single_class(
                str(raw_path), str(refined_path), str(discard_path),
                mode, n, threshold, batch_size
            )

            print(f"\n单类精炼汇总:")
            print(f"  类别: {class_name}")
            print(f"  保留: {refined_count} 张")
            print(f"  丢弃: {discard_count} 张")

        elif subdirs:
            # 多类数据：包含多个子目录
            print(f"检测到多类数据模式，共 {len(subdirs)} 个类别")

            total_refined = 0
            total_discarded = 0

            for class_dir in tqdm(subdirs, desc="处理各类别"):
                class_name = class_dir.name
                refined_path = refined_base_path / class_name
                discard_path = discard_base_path / class_name

                print(f"\n处理类别: {class_name}")
                refined_count, discard_count = self.refine_single_class(
                    str(class_dir), str(refined_path), str(discard_path),
                    mode, n, threshold, batch_size
                )

                total_refined += refined_count
                total_discarded += discard_count

            print(f"\n多类精炼汇总:")
            print(f"  总类别数: {len(subdirs)}")
            print(f"  总保留: {total_refined} 张")
            print(f"  总丢弃: {total_discarded} 张")

        else:
            print(f"在 {raw_path} 中没有找到图像文件或子目录")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='基于DINOv3特征的数据精炼')

    # 路径参数
    parser.add_argument('--raw_path', type=str, required=True,
                        help='原始数据路径')
    parser.add_argument('--refined_path', type=str, required=True,
                        help='精炼后数据保存路径')
    parser.add_argument('--discard_path', type=str, required=True,
                        help='丢弃数据保存路径')

    # 精炼模式参数
    parser.add_argument('--mode', type=str, required=True,
                        choices=['nearest', 'remove_farthest', 'threshold'],
                        help='精炼模式: nearest-保留最近的n个, remove_farthest-去除最远的n个, threshold-基于阈值')

    # 模式相关参数
    parser.add_argument('--n', type=int, default=None,
                        help='模式1和2的样本数量')
    parser.add_argument('--threshold', type=float, default=None,
                        help='模式3的距离阈值')

    # 其他参数
    parser.add_argument('--batch_size', type=int, default=8,
                        help='批次大小')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cuda:0', 'cuda:1', 'cpu'],
                        help='设备类型')

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    print("数据精炼参数:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")

    # 参数验证
    if args.mode in ['nearest', 'remove_farthest'] and args.n is None:
        raise ValueError(f"模式 {args.mode} 需要指定 --n 参数")
    if args.mode == 'threshold' and args.threshold is None:
        raise ValueError("模式 threshold 需要指定 --threshold 参数")

    # 创建精炼器
    refiner = DINOv3DataRefiner(device=args.device)

    # 执行精炼
    refiner.refine_data(
        raw_path=args.raw_path,
        refined_base_path=args.refined_path,
        discard_base_path=args.discard_path,
        mode=args.mode,
        n=args.n,
        threshold=args.threshold,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()
