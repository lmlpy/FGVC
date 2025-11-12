import os.path
import time
import torch.nn.functional as F
import torch
from modelscope import AutoImageProcessor, AutoModel
from transformers.image_utils import load_image

class DINOvMultiLayerFeatureExtractor(torch.nn.Module):
    def __init__(self, model=None, layer_names=[-1], device='cuda'):
        super().__init__()
        self.model = model
        self.device = device
        self.layer_names = layer_names
        self.features = {}
    def load_dinov3(self, pretrained_model_name = "facebook/dinov3-vits16-pretrain-lvd1689m", cache_dir = '/root/.cache/modelscope/hub/models'):

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
        outputs = self.model(x)
        all_hidden_states = outputs.hidden_states  # 长度为 num_layers + 1（包含embedding层）
        #print(len(all_hidden_states))
        #print([i.shape for i in all_hidden_states])
        # 示例：选择特定层（如第3、6、9层
        cls_token = all_hidden_states[-1][:, 0, :]
        #print([i.shape for i in multi_scale_features])

        return cls_token
if __name__ == "__main__":
    pass