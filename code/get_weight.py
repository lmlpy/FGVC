import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights
import warnings
import models_dinov2
from collections import OrderedDict
import os

def main(backbone_path='../weight/backbone-l.pth', preweight_path='../weight/pre_weight.pth', model='vit_large'):
        import_student = getattr(models_dinov2, model)
        encoder = import_student(img_size=224,
                           patch_size=14,
                           init_values=1.0,
                           ffn_layer='mlp',
                           block_chunks=0,
                           num_register_tokens=0,
                           interpolate_antialias=False,
                           interpolate_offset=0.1).to('cuda:0')
        checkpoint = torch.load(backbone_path, weights_only=False, map_location='cpu')
        print(checkpoint.keys())
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'teacher' in checkpoint:
            state_dict = checkpoint['teacher']
        else:
            state_dict = checkpoint

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if 'student' in k:
                new_state_dict[k.replace("student.backbone.", "")] = v
        if not os.path.exists(preweight_path):
            torch.save(new_state_dict, preweight_path)
        msg = encoder.load_state_dict(new_state_dict, strict=False)
        print(msg)

if __name__ == "__main__":
    main()
