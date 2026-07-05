import warnings

import torch.nn as nn
import torch
from torchvision import models

from .model_utils import *

def make_model(arch, num_classes=1000, weights=None, multi_crop_wrapper=False, input_res='high', relu_inplance=False):

    ## base model
    model = models.__dict__[arch](weights=weights)

    if arch.startswith('resnet'):
        ## overwite forward function of resnet model
        model = resnet_overwrite_forward(model,
                                         input_res=input_res)
      
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
    
    if multi_crop_wrapper:
        return MultiCropWrapper(model)
    else:
        return model

def make_frank_model(arch, 
                     front_hook_post_relu,
                     end_hook_post_relu,
                     init_transform,
                     K_init,
                     bn_layers,
                     num_classes=1000, 
                     front_layer_index=1,
                     stitch_layer_index=1,
                     weights_front=None, 
                     weights_end=None,
                     weights_frank=None,
                     multi_crop_wrapper=False,
                     data_loader=None,
                     input_res='high'):
    
    
    assert weights_front is not None, "weights_front should not be None."
    assert weights_end is not None, "weights_end should not be None." 

    
    front_model = make_model(arch, num_classes=num_classes,
                             multi_crop_wrapper=False,
                             input_res=input_res)    
    front_model = load_model(front_model, torch.load(weights_front, 
                                                     weights_only=False)['model']) 

    end_model = make_model(arch, num_classes=num_classes, 
                           multi_crop_wrapper=False,
                           input_res=input_res)
    end_model = load_model(end_model, torch.load(weights_end,
                                                 weights_only=False)['model'])

    if stitch_layer_index == 0:
        
        model = end_model
    
    elif stitch_layer_index == get_stitch_num_layers(arch)+1:

        model = front_model
   
    else:
        
        model = FrankModel(arch, 
                           front_model, end_model,
                           front_hook_post_relu=front_hook_post_relu,
                           end_hook_post_relu=end_hook_post_relu,
                           front_layer_index=front_layer_index,
                           stitch_layer_index=stitch_layer_index,
                           init_transform=init_transform,
                           K_init=K_init,
                           bn_layers=bn_layers,
                           data_loader=data_loader)
        
        if weights_frank:
            model = load_model(model, torch.load(weights_frank)['model'])

    if multi_crop_wrapper:
        return MultiCropWrapper(model)
    else:
        return model    


