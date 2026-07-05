import warnings

import torch.nn as nn
import torch

from dataloaders.multi_crop import MultiCropWrapper
from .model_utils import *


def make_model(arch, num_classes=1000, weights=None, multi_crop_wrapper=False, input_res='high', relu_inplance=False, remove_residuals=False):


    if not(arch.startswith('resnet')) and remove_residuals:
        raise ValueError("Removing residual connection is only relevant for resnet architectures.")


    if arch.startswith('resnet'):
        
        from .resnet import get_resnet_models


        model = get_resnet_models(arch, remove_residuals=remove_residuals,
                                  input_res=input_res, relu_inplance=relu_inplance,
                                  num_classes=num_classes, weights=None) ## TODO: remove support for weight loading

    elif arch.startswith('vgg'):

        from .vgg import get_vgg_models


        model = get_vgg_models(arch, num_classes=num_classes, num_channels=3) 


    elif arch.startswith('vit'):

        from .vit import get_vit_models

        model = get_vit_models(arch, num_classes=num_classes, num_channels=3) ## TODO: add support for weight loading
        
    
    if weights:
        model = load_model(model, torch.load(weights, weights_only=False)['model'])


    if multi_crop_wrapper:
        return MultiCropWrapper(model)
    else:
        return model
    
    
def make_simclr_model(arch, num_classes=1000, weights=None, multi_crop_wrapper=False, input_res='high', relu_inplance=False,
                      remove_residuals=False,
                      projection_dim=128):

    if arch.startswith('resnet'):
        
        from .resnet import get_resnet_models


        model = get_resnet_models(arch, remove_residuals=remove_residuals,
                                  input_res=input_res, relu_inplance=relu_inplance,
                                  num_classes=num_classes, weights=None) ## TODO: remove support for weight loading
        
         
        projector = nn.Sequential(nn.Linear(model.fc.in_features, 1024),
                                  nn.ReLU(),
                                  nn.Linear(1024, projection_dim))


    elif arch.startswith('vgg'):

        from .vgg import get_vgg_models


        model = get_vgg_models(arch, num_classes=num_classes, num_channels=3) 

        projector = nn.Sequential(nn.Linear(model.classifier[0].in_features, 1024),
                                  nn.ReLU(),
                                  nn.Linear(1024, projection_dim))
        

    elif arch.startswith('vit'):

        from .vit import get_vit_models

        model = get_vit_models(arch, num_classes=num_classes, num_channels=3) ## TODO: add support for weight loading

        projector = nn.Sequential(nn.Linear(model.head.in_features, 1024),
                                  nn.ReLU(),
                                  nn.Linear(1024, projection_dim))

        

    else:
        raise ValueError("Autoencoder baseline only supports ResNet architectures.")

    if multi_crop_wrapper:
        model = MultiCropWrapper(model)
    
    return model, projector


def make_frank_model(arch_front, 
                     arch_end,
                     front_hook_post_relu,
                     end_hook_post_relu,
                     init_transform_mode,
                     K_init,
                     bn_layers,
                     front_layer_index=1,
                     stitch_layer_index=1,
                     relative_front_index=False,                 
                     weights_front=None, 
                     weights_end=None,
                     weights_frank=None,
                     multi_crop_wrapper=False,
                     data_loader=None,
                     input_res='high',
                     remove_residuals_front=False,
                     remove_residuals_end=False,
                     num_classes_front=1000, 
                     num_classes_end=1000,):
    
    
    assert weights_front is not None, "weights_front should not be None."
    assert weights_end is not None, "weights_end should not be None." 

    
    front_model = make_model(arch_front, num_classes=num_classes_front,
                             multi_crop_wrapper=False,
                             input_res=input_res,
                             remove_residuals=remove_residuals_front)    
    front_model = load_model(front_model, torch.load(weights_front, 
                                                     weights_only=False)['model']) 

    end_model = make_model(arch_end, num_classes=num_classes_end, 
                           multi_crop_wrapper=False,
                           input_res=input_res,
                           remove_residuals=remove_residuals_end)
    end_model = load_model(end_model, torch.load(weights_end,
                                                 weights_only=False)['model'])

        
    model = FrankModel(arch_front, arch_end,
                       front_model, end_model,
                       front_hook_post_relu=front_hook_post_relu,
                       end_hook_post_relu=end_hook_post_relu,
                       front_layer_index=front_layer_index,
                       stitch_layer_index=stitch_layer_index,
                       relative_front_index=relative_front_index,
                       init_transform_mode=init_transform_mode,
                       K_init=K_init,
                       bn_layers=bn_layers,
                       data_loader=data_loader,)
        
    if weights_frank:
        model = load_model(model, torch.load(weights_frank)['model'])


    if multi_crop_wrapper:
        return MultiCropWrapper(model)
    else:
        return model    