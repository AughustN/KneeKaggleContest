# Vendored DINOv2 from facebookresearch/dinov2 (Apache 2.0)
# Minimal inference-only build — no xformers, no torchvision dependency

import math
import torch
from knee.dinov2.vision_transformer import vit_small, DinoVisionTransformer

__all__ = ["vit_small", "DinoVisionTransformer", "load_pretrained"]


def load_pretrained(model, pretrained_path, strict=False):
    """Load pretrained DINOv2 weights with automatic pos_embed interpolation.

    Handles the img_size mismatch: pretrained weights are from img_size=518
    (1369 patches), but our model uses img_size=224 (256 patches).
    The pos_embed is interpolated via bicubic to match.
    """
    sd = torch.load(pretrained_path, map_location="cpu", weights_only=True)

    # Interpolate pos_embed if shape mismatches
    if "pos_embed" in sd and "pos_embed" in model.state_dict():
        target_shape = model.state_dict()["pos_embed"].shape
        src_shape = sd["pos_embed"].shape
        if src_shape != target_shape:
            cls_token = sd["pos_embed"][:, :1]
            patch_pos = sd["pos_embed"][:, 1:]
            N_src = patch_pos.shape[1]
            M_src = int(math.sqrt(N_src))
            N_target = target_shape[1] - 1
            M_target = int(math.sqrt(N_target))
            dim = patch_pos.shape[2]
            patch_pos_2d = patch_pos.reshape(1, M_src, M_src, dim).permute(0, 3, 1, 2)
            patch_pos_interp = torch.nn.functional.interpolate(
                patch_pos_2d, size=(M_target, M_target), mode="bicubic", antialias=True,
            )
            patch_pos_interp = patch_pos_interp.permute(0, 2, 3, 1).reshape(1, -1, dim)
            sd["pos_embed"] = torch.cat([cls_token, patch_pos_interp], dim=1)

    model.load_state_dict(sd, strict=strict)
    return model
