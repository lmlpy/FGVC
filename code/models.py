import torch
import torch.nn as nn
import warnings
import models_dinov2
from dinov import DINOvMultiLayerFeatureExtractor
from collections import OrderedDict

warnings.filterwarnings('ignore')

class SimpleNet(nn.Module):
    """
    基于ViT-B/16的细粒度分类网络
    """

    def __init__(self, num_classes: int = 10, input_channels: int = 3, pretrained: bool = True, device: str = 'cuda:1'):
        """
        初始化SimpleNet

        Args:
            num_classes (int): 分类类别数
            pretrained (bool): 是否使用ImageNet预训练权重
        """
        super(SimpleNet, self).__init__()

        self.num_classes = num_classes
        weight_path = '../weight/pre_weight.pth'
        #weight_path = '../code-v5-dino/backbone-l.pth'
        try:
            import_student = getattr(models_dinov2, 'vit_large')
            self.encoder = import_student(img_size=224,
                               patch_size=14,
                               init_values=1.0,
                               ffn_layer='mlp',
                               block_chunks=0,
                               num_register_tokens=0,
                               interpolate_antialias=False,
                               interpolate_offset=0.1)
            state_dict = torch.load(weight_path, weights_only=False, map_location='cpu')
            self.encoder.load_state_dict(state_dict, strict=True)
            self.encoder.to(device)
            print("ImageNet 1k")
        except:
            self.encoder = DINOvMultiLayerFeatureExtractor(layer_names=[-1], device=device)
            self.encoder.load_dinov3(pretrained_model_name="facebook/dinov3-vitl16-pretrain-lvd1689m")
            state_dict = torch.load(weight_path, weights_only=False, map_location='cpu')
            self.encoder.model.load_state_dict(state_dict, strict=True)

        self.dim = 1024

        # 替换分类头以适应细粒度分类任务
        self.heads = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.dim , 1024),
            nn.GELU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.2),
            nn.Linear(1024, num_classes)
        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.gmp = nn.AdaptiveMaxPool2d((1, 1))

        # 冻结部分层进行微调（可选）
        self._freeze_layers()

    def _freeze_layers(self, num_unfrozen_blocks=0):
        """解冻最后N个blocks"""
        # 全部冻结
        for param in self.encoder.parameters():
            param.requires_grad = False

        # 解冻最后N个blocks
        for name, param in self.encoder.named_parameters():
            if '.layer.' in name:
                # 提取block编号
                block_num = int(name.split('.layer.')[1].split('.')[0])
                if block_num >= 24 - num_unfrozen_blocks:  # DINOv3-small有12个blocks
                    print(name)
                    param.requires_grad = True

        # 分类头可训练
        for param in self.heads.parameters():
            param.requires_grad = True

        print(f"冻结策略: 解冻最后{num_unfrozen_blocks}个blocks")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x (torch.Tensor): 输入张量 [batch_size, 3, 224, 224]

        Returns:
            torch.Tensor: 分类logits [batch_size, num_classes]
        """
        cls_token = self.encoder(x)
        #print(f"Patch token {[patch.shape for patch in patch_token]}")
        #print(f"CLS token {[cls.shape for cls in cls_token]}")
        #x = self.heads(torch.cat([cls_token, patch_token], dim=-1))
        x = self.heads(cls_token)
        return x

    def unfreeze_all(self):
        """
        解冻所有层（在训练后期使用）
        """
        for param in self.vit.parameters():
            param.requires_grad = True
        print("解冻所有层进行全网络微调")


# 使用示例
if __name__ == "__main__":
    # 创建模型
    model = SimpleNet(num_classes=100, pretrained=True, device='cuda:0').to('cuda:0')

    # 测试前向传播
    x = torch.randn(2, 3, 224, 224).cuda()
    output = model(x)
    print(f"输入尺寸: {x.shape}")
    print(f"输出尺寸: {output.shape}")
    print(f"输出范围: [{output.min():.4f}, {output.max():.4f}]")

    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    print(f"训练比例: {trainable_params / total_params * 100:.2f}%")
