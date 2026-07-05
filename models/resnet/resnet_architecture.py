import torch
import torch.nn as nn
from torchvision import models

import types
from functools import partial

import warnings


def resnet_overwrite_forward(model, input_res='high', remove_residuals=False):
    # Overwrite forwards methods to facilitate hooking

    class Add(nn.Module):
        def __init__(self, scale_x=1.0, scale_y=1.0):
            super(Add, self).__init__()
            self.scale_x = scale_x
            self.scale_y = scale_y

        def forward(self, x, y):
            return self.scale_x * x + self.scale_y * y
        
    class Identity(nn.Module):
        def forward(self, x):
            return x
        
    def _forward_main_high(self, x, return_feat=None):
        x = self.conv1(x)
        x = self.bn1(x)
        
        x = self.pre_act(x)    
        x = self.relu(x)
        x = self.post_act(x)

        x = self.maxpool(x)


        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        feat_map = x 
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        feat = x

        x = self.fc(x)

        if return_feat == 'feat_map':
            return x, feat_map
        elif return_feat == 'feat':
            return x, feat
        else:
            return x

    def _forward_main_low(self, x, return_feat=None):

        x = self.conv1(x)
        x = self.bn1(x)
        
        x = self.pre_act(x)
        x = self.relu(x)
        x = self.post_act(x)


        # x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        feat_map = x 
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        feat = x

        x = self.fc(x)

        if return_feat == 'feat_map':
            return x, feat_map
        elif return_feat == 'feat':
            return x, feat
        else:
            return x

    def _forward_bottleneck(self, x, with_residual=True):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = self.residual_identity(out)

        if with_residual:
            if self.downsample is not None:
                identity = self.downsample(identity)
            
            identity = self.shortcut_identity(identity)
            out = self.add(out, identity)

        out = self.add_pre_act(out)
        out = self.relu(out)
        out = self.add_post_act(out)

        return out

    def _forward_basic_block(self, x, with_residual=True):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.residual_identity(out) 

        if with_residual:     
            if self.downsample is not None:
                identity = self.downsample(x)

            identity = self.shortcut_identity(identity) 

            
            out = self.add(out, identity)

        out = self.add_pre_act(out)
        out = self.relu(out)
        out = self.add_post_act(out)

        return out
    
    ## Overwrite forward method of ResNet class
    model.pre_act = Identity()        
    model.post_act = Identity()


    if input_res == 'high':
        model.forward = _forward_main_high.__get__(model, model.__class__)
    
    else: 
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False) # NOTE you change the stride, padding and kernel size of the original resnet!
        model.forward = _forward_main_low.__get__(model, model.__class__) # NOTE Replaces the forward. BUT DOES NOT CALL IT


    ## Overwrite forward method of Bottleneck class
    for _, module in model.named_modules():
        if type(module).__name__ == 'Bottleneck':
            module.add = Add()
            
            module.add_pre_act = Identity()
            module.add_post_act = Identity()

            module.shortcut_identity = Identity()
            module.residual_identity = Identity()

            module.forward = partial(_forward_bottleneck.__get__(module, module.__class__), with_residual=not(remove_residuals))

        elif type(module).__name__ == 'BasicBlock': 
            module.add = Add()

            module.add_post_act = Identity()
            module.add_pre_act = Identity()

            module.shortcut_identity = Identity()
            module.residual_identity = Identity()

            module.forward = partial(_forward_basic_block.__get__(module, module.__class__), with_residual=not(remove_residuals))

    return model

def resnet_feature_map_hooks(model):

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

def resnet_get_feature_maps(model):

    def _get_feature_map(self, layer_index):

        key = list(self.hooked_activations.keys())[layer_index]
        feature_map = self.hooked_activations[key]
        
        return feature_map

    model.get_feature_map = types.MethodType(_get_feature_map, model)

    return model

def get_classifier(model):
    return model.fc

def get_resnet_models(model_name: str, remove_residuals: bool,  
                      input_res: str, relu_inplance: bool, num_classes: int, weights=None) -> nn.Module:

    def _remove_resnet_downsamples(model):
        for _, module in model.named_modules():
            if isinstance(module, nn.Sequential):
                for _, sub_module in module.named_children():
                    if isinstance(sub_module, nn.Module) and hasattr(sub_module, "downsample"):
                        if 'downsample' in sub_module._modules:
                            #print(f"Removing downsample from {name}.{sub_name}")
                            sub_module.downsample = None  # This disables usage          
                            #Also remove from _modules to avoid parameter count
                            del sub_module._modules['downsample']
                            # print("Removed downsample from module")
        return model



    model = models.__dict__[model_name](weights=weights) 

    if remove_residuals:
            model = _remove_resnet_downsamples(model)

    ## add get classifier function
    model.get_classifier = types.MethodType(get_classifier, model)

    ## overwite forward function of resnet model
    model = resnet_overwrite_forward(model,
                                    input_res=input_res, 
                                    remove_residuals=remove_residuals)
    
    model = resnet_feature_map_hooks(model)
    model = resnet_get_feature_maps(model)
    
    ## modify relu inplace parameter
    for module in model.modules():
        if isinstance(module, nn.ReLU):
            module.inplace = relu_inplance

    if num_classes != model.fc.out_features:
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    if weights:
        warnings.warn("Changing the number of classes in the model while using the pretrained weights.")


    return model


