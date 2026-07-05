import torch
import torch.nn as nn

from timm.models.vision_transformer import VisionTransformer
from timm.models._manipulate import checkpoint_seq
import types

def vit_tiny_patch4_32(num_classes: int, num_channels: int, **kwargs):
    return VisionTransformer(num_classes=num_classes,
                             in_chans=num_channels,
                             img_size=32,
                             patch_size=4,
                             embed_dim=192,
                             depth=12,
                             num_heads=3,
                             **kwargs)


def vit_small_patch4_32(num_classes: int, num_channels: int, **kwargs):
    return VisionTransformer(num_classes=num_classes,
                             in_chans=num_channels,
                             img_size=32,
                             patch_size=4,
                             embed_dim=384,
                             depth=12,
                             num_heads=6,
                             **kwargs)


def vit_medium_patch4_32(num_classes: int, num_channels: int, **kwargs):
    return VisionTransformer(num_classes=num_classes,
                             in_chans=num_channels,
                             img_size=32,
                             patch_size=4,
                             embed_dim=512,
                             depth=12,
                             num_heads=8,
                             **kwargs)


def vit_base_patch4_32(num_classes: int, num_channels: int, **kwargs):
    return VisionTransformer(num_classes=num_classes,
                             in_chans=num_channels,
                             img_size=32,
                             patch_size=4,
                             embed_dim=768,
                             depth=12,
                             num_heads=12,
                             **kwargs)

def vit_large_patch4_32(num_classes: int, num_channels: int, **kwargs):
    return VisionTransformer(num_classes=num_classes,
                             in_chans=num_channels,
                             img_size=32,
                             patch_size=4,
                             embed_dim=1024,
                             depth=24,
                             num_heads=16,
                             **kwargs)


def vit_huge_patch4_32(num_classes: int, num_channels: int, **kwargs):
    return VisionTransformer(num_classes=num_classes,
                             in_chans=num_channels,
                             img_size=32,
                             patch_size=4,
                             embed_dim=1280,
                             depth=32,
                             num_heads=16,
                             **kwargs)


def vit_giant_patch4_32(num_classes: int, num_channels: int, **kwargs):
    return VisionTransformer(num_classes=num_classes,
                             in_chans=num_channels,
                             img_size=32,
                             patch_size=4,
                             embed_dim=1408,
                             depth=40,
                             num_heads=16,
                             **kwargs)


def vit_gigantic_patch4_32(num_classes: int, num_channels: int, **kwargs):
    return VisionTransformer(num_classes=num_classes,
                             in_chans=num_channels,
                             img_size=32,
                             patch_size=4,
                             embed_dim=1664,
                             depth=48,
                             num_heads=16,
                             **kwargs)


def vit_tiny_patch16_224(num_classes: int, num_channels: int, **kwargs):
    return VisionTransformer(num_classes=num_classes,
                             in_chans=num_channels,
                             img_size=224,
                             patch_size=16,
                             embed_dim=192,
                             depth=12,
                             num_heads=3,
                             **kwargs)


def vit_small_patch16_224(num_classes: int, num_channels: int, **kwargs):
    return VisionTransformer(num_classes=num_classes,
                             in_chans=num_channels,
                             img_size=224,
                             patch_size=16,
                             embed_dim=384,
                             depth=12,
                             num_heads=6,
                             **kwargs)


def vit_medium_patch16_224(num_classes: int, num_channels: int, **kwargs):
    return VisionTransformer(num_classes=num_classes,
                             in_chans=num_channels,
                             img_size=224,
                             patch_size=16,
                             embed_dim=512,
                             depth=12,
                             num_heads=8,
                             **kwargs)


def vit_base_patch16_224(num_classes: int, num_channels: int, **kwargs):
    return VisionTransformer(num_classes=num_classes,
                             in_chans=num_channels,
                             img_size=224,
                             patch_size=16,
                             embed_dim=768,
                             depth=12,
                             num_heads=12,
                             **kwargs)

def vit_large_patch16_224(num_classes: int, num_channels: int, **kwargs):
    return VisionTransformer(num_classes=num_classes,
                             in_chans=num_channels,
                             img_size=224,
                             patch_size=16,
                             embed_dim=1024,
                             depth=24,
                             num_heads=16,
                             **kwargs)


def vit_huge_patch16_224(num_classes: int, num_channels: int, **kwargs):
    return VisionTransformer(num_classes=num_classes,
                             in_chans=num_channels,
                             img_size=224,
                             patch_size=16,
                             embed_dim=1280,
                             depth=32,
                             num_heads=16,
                             **kwargs)


def vit_giant_patch16_224(num_classes: int, num_channels: int, **kwargs):
    return VisionTransformer(num_classes=num_classes,
                             in_chans=num_channels,
                             img_size=224,
                             patch_size=16,
                             embed_dim=1408,
                             depth=40,
                             num_heads=16,
                             **kwargs)


def vit_gigantic_patch16_224(num_classes: int, num_channels: int, **kwargs):
    return VisionTransformer(num_classes=num_classes,
                             in_chans=num_channels,
                             img_size=224,
                             patch_size=16,
                             embed_dim=1664,
                             depth=48,
                             num_heads=16,
                             **kwargs)

def _block_forward(self, x: torch.Tensor) -> torch.Tensor:
    

    ## TODO: remove V
    # norm1_x = self.norm1(x)
    # B, N, C = norm1_x.shape
    # qkv = self.attn.qkv(norm1_x)
    # x_shape = x.shape[2]
    
    # xqkv = torch.cat((x, qkv), dim=2)  ## TODO: comment in this line
    # xqkv = self.xqkv(xqkv)  ## TODO: comment in this line

    # x, qkv = xqkv[..., :x_shape], xqkv[..., x_shape:]
    # x = x + self.drop_path1(self.ls1(self.attn.forward_with_qkv(qkv, B, N, C))) 
    # # ##### TODO: remove ^



    x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    
    return x

def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
    x = self.patch_embed(x)
    x = self._pos_embed(x)
    x = self.patch_drop(x)
    x = self.norm_pre(x)
    if self.grad_checkpointing and not torch.jit.is_scripting():
        x = checkpoint_seq(self.blocks, x)
    else:

        for block_index, block in enumerate(self.blocks):
            x = block(x)
        # x = self.blocks(x)

    x = self.norm(x)
    return x

def _forward_head(self, x: torch.Tensor, pre_logits: bool = False, return_feat = None) -> torch.Tensor:
    feat_map = x 
    x = self.pool(x)
    x = self.fc_norm(x)
    x = self.head_drop(x)
    feat = x

    if return_feat == 'feat_map':
        return x if pre_logits else self.head(x), feat_map
    elif return_feat == 'feat':
        return x if pre_logits else self.head(x), feat
    else:
        return x if pre_logits else self.head(x)

def _forward(self, x: torch.Tensor, return_feat=None) -> torch.Tensor:
    x = self.forward_features(x)
    
    if return_feat:
        x, feat = self.forward_head(x, return_feat=return_feat)
        return x, feat
    else:
        x = self.forward_head(x)
        return x

def vit_overwrite_forward(model: nn.Module) -> nn.Module:
    # Overwrite forwards methods to facilitate hooking
    model.forward_features = _forward_features.__get__(model, VisionTransformer)
    model.forward_head = _forward_head.__get__(model, VisionTransformer)
    model.forward = _forward.__get__(model, VisionTransformer)

    return model


def vit_feature_map_hooks(model):

    def add_hooks(self, layers_to_hook):

        def _activation_hook(name, module, input, output):
            self.hooked_activations[name] = output 

            self.hooked_activations[name].requires_grad_()
            self.hooked_activations[name].retain_grad()

        for name, module in self.named_modules():
            if name in layers_to_hook:
                # module.register_forward_hook(_activation_hook)
                module.register_forward_hook(lambda module, input, output, name=name: _activation_hook(name, module, input, output))

        return
    
    model.hooked_activations = {}
    model.add_hooks = types.MethodType(add_hooks, model)

    return model

def get_classifier(model):

    return model.head


def get_vit_models(model_name: str, num_classes: int, num_channels: int) -> nn.Module:
    if model_name == "vit_tiny_patch4_32":
        model=vit_tiny_patch4_32(num_classes, num_channels)
    if model_name == "vit_small_patch4_32":
        model=vit_small_patch4_32(num_classes, num_channels)
    if model_name == "vit_medium_patch4_32":
        model=vit_medium_patch4_32(num_classes, num_channels)
    if model_name == "vit_base_patch4_32":
        model=vit_base_patch4_32(num_classes, num_channels)
    if model_name == "vit_large_patch4_32":
        model=vit_large_patch4_32(num_classes, num_channels)
    if model_name == "vit_huge_patch4_32":
        model=vit_huge_patch4_32(num_classes, num_channels)
    if model_name == "vit_giant_patch4_32":
        model=vit_giant_patch4_32(num_classes, num_channels)
    if model_name == "vit_gigantic_patch4_32":
        model=vit_gigantic_patch4_32(num_classes, num_channels)
    if model_name == "vit_tiny_patch16_224":
        model=vit_tiny_patch16_224(num_classes, num_channels)
    if model_name == "vit_small_patch16_224":
        model=vit_small_patch16_224(num_classes, num_channels)
    if model_name == "vit_medium_patch16_224":
        model=vit_medium_patch16_224(num_classes, num_channels)
    if model_name == "vit_base_patch16_224":
        model=vit_base_patch16_224(num_classes, num_channels)
    if model_name == "vit_large_patch16_224":
        model=vit_large_patch16_224(num_classes, num_channels)
    if model_name == "vit_huge_patch16_224":
        model=vit_huge_patch16_224(num_classes, num_channels)
    if model_name == "vit_giant_patch16_224":
        model=vit_giant_patch16_224(num_classes, num_channels)
    if model_name == "vit_gigantic_patch16_224":
        model=vit_gigantic_patch16_224(num_classes, num_channels)



    # Overwrite the block forward method to facilitate hooking
    for block in model.blocks:
        block.xqkv = nn.Identity()
        block.forward = _block_forward.__get__(block, VisionTransformer)

    model = vit_overwrite_forward(model)

    ## add get classifier function
    model.get_classifier = types.MethodType(get_classifier, model)

    model = vit_feature_map_hooks(model)

    return model