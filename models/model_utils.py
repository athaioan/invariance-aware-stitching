from importlib import util
import torch.nn as nn
import torch
from functools import reduce, partial
from .stitch import STITCH_MAP


import numpy as np

from utils import freeze_bn_running_stats, unfreeze_bn_running_stats


def rgetattr(obj, attr, *args):
    ''' get attribute recursively, eg. self.layer0.conv.weight '''

    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return reduce(_getattr, [obj] + attr.split('.'))

def get_stitch_num_layers(arch):

    if arch.startswith('vit'):
        num_stitch_layers = len(get_stitch_layer_name(arch))

    elif arch.startswith('resnet'):
        num_stitch_layers = len(get_stitch_layer_name(arch))

    elif arch.startswith('vgg'):
        num_stitch_layers = len(get_stitch_layer_name(arch))

    else:
        raise ValueError("Only resnet and vit are supported for now")
   
    return num_stitch_layers

def get_stitch_layer_name(arch, arch_end=None, post_act=False, relative_front_index=False):
    # Function to get the layer name from the stitch layer index

    def _get_arch_layers(arch, post_act):

        if arch.startswith('resnet'):
            if post_act:
                in_block_prefix_mode = 'add_post_act'
                out_block_prefix_mode = 'post_act'
            else:
                in_block_prefix_mode = 'add_pre_act'
                out_block_prefix_mode = 'pre_act'

        if arch == 'resnet50':

            arch_layers = {1: f'{out_block_prefix_mode}',
                           2: f'layer1.0.{in_block_prefix_mode}',
                           3: f'layer1.2.{in_block_prefix_mode}',
                           4: f'layer2.0.{in_block_prefix_mode}',
                           5: f'layer2.3.{in_block_prefix_mode}',
                           6: f'layer3.0.{in_block_prefix_mode}',
                           7: f'layer3.5.{in_block_prefix_mode}', 
                           8: f'layer4.0.{in_block_prefix_mode}',
                           9: f'layer4.2.{in_block_prefix_mode}',}
            
        elif arch == 'resnet34':

            arch_layers =  {1: f'{out_block_prefix_mode}',
                            2: f'layer1.0.{in_block_prefix_mode}',
                            3: f'layer1.2.{in_block_prefix_mode}',
                            4: f'layer2.0.{in_block_prefix_mode}',
                            5: f'layer2.3.{in_block_prefix_mode}',
                            6: f'layer3.0.{in_block_prefix_mode}',
                            7: f'layer3.5.{in_block_prefix_mode}', 
                            8: f'layer4.0.{in_block_prefix_mode}',
                            9: f'layer4.2.{in_block_prefix_mode}',}

        elif arch == 'resnet18':

            arch_layers =  {1: f'{out_block_prefix_mode}',
                            2: f'layer1.0.{in_block_prefix_mode}',
                            3: f'layer1.1.{in_block_prefix_mode}',
                            4: f'layer2.0.{in_block_prefix_mode}',
                            5: f'layer2.1.{in_block_prefix_mode}',
                            6: f'layer3.0.{in_block_prefix_mode}',
                            7: f'layer3.1.{in_block_prefix_mode}', 
                            8: f'layer4.0.{in_block_prefix_mode}',
                            9: f'layer4.1.{in_block_prefix_mode}',}

        elif arch == 'vgg16':

            arch_layers = {1: f'features.2',
                           2: f'features.5',
                           3: f'features.12',
                           4: f'features.22',
                           5: f'features.32',
                           6: f'features.42',}

        elif arch == 'vit_tiny_patch4_32':

            arch_layers = {1: f'blocks.0',
                           2: f'blocks.1',
                           3: f'blocks.2',
                           4: f'blocks.3',
                           5: f'blocks.4',
                           6: f'blocks.5',
                           7: f'blocks.6',
                           8: f'blocks.7',
                           9: f'blocks.8',
                          10: f'blocks.9',
                          11: f'blocks.10',
                          12: f'blocks.11',
                          13: f'head_drop',}
                          
        elif arch == 'vit_base_patch16_224':

            arch_layers = {1: f'blocks.0',
                           2: f'blocks.2',
                           3: f'blocks.4',
                           4: f'blocks.6',
                           5: f'blocks.8',
                           6: f'blocks.10',
                           7: f'head_drop',}

        else:
            raise ValueError("Only resnet50, resnet34, resnet18 and vit_tiny_patch4_32 are supported for now")
        
        return arch_layers
    
    if relative_front_index and arch_end is None:
        raise ValueError("relative_front_index is not supported when arch_end is not provided.")

    if relative_front_index:
        arch_layers =  _get_arch_layers(arch, post_act)
        arch_end_layers = _get_arch_layers(arch_end, post_act)
        arch_to_arch_end_mapping = np.linspace(1, len(arch_layers), len(arch_end_layers)).round().astype(int)        
        arch_layers = {k: arch_layers[v] for k, v in zip(arch_end_layers.keys(), arch_to_arch_end_mapping)}

    else:
        arch_layers =  _get_arch_layers(arch, post_act)

    return arch_layers


    

def get_stitch_transform(front_activations=None,
                         end_activations=None,
                         bn_layers=False,
                         init_mode='rand',):
    

    def _get_representation_format(representation_dim):

        if len(representation_dim) == 4:
            return 'conv'
        
        elif len(representation_dim) == 3:
            return 'token'
        
        elif len(representation_dim) == 2:
            return 'vector'
          
          
    front_format = _get_representation_format(front_activations.shape)
    end_format = _get_representation_format(end_activations.shape)

    key = (front_format, end_format)

    if key not in STITCH_MAP:
        raise ValueError(f"Unsupported representation format: {front_format} to {end_format} for stitching.")
    
    stitch_cls, needs_init = STITCH_MAP[key]
    kwargs = dict(with_bn=bn_layers)
    if needs_init:
        kwargs['init_mode'] = init_mode

    stitch_transform = stitch_cls(front_activations, end_activations, **kwargs)

    return stitch_transform


def load_model(model, weight_dict):
    
    # remove 'module.model.' prefix from the keys of the state_dict
    new_weight_dict = {}
    for k, v in weight_dict.items():
        if k.startswith('module.model.'):
            new_weight_dict[k[13:]] = v
        else:
            new_weight_dict[k] = v

    # model_dict = model.state_dict()
    # new_weight_dict = {k: v for k, v in new_weight_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
    # model_dict.update(new_weight_dict) 
    # model.load_state_dict(model_dict)

    model.load_state_dict(new_weight_dict)

    return model


class FrankModel(nn.Module):
    """
    Perform forward pass within a stitched model.
    """

    def __init__(self, 
                 arch_front,arch_end, 
                 front_model, end_model, 
                 front_hook_post_relu=False,
                 end_hook_post_relu=False,
                 front_layer_index=1,
                 stitch_layer_index=1,
                 relative_front_index=False,                 
                 init_transform_mode='rand',
                 K_init=100,
                 bn_layers=False,
                 data_loader=None,):

        super().__init__()

        if stitch_layer_index == 0:
            self.stitch_mode = 'end'
            self.end_model = end_model
        elif stitch_layer_index == get_stitch_num_layers(arch_end) + 1:
            self.stitch_mode = 'front'
            self.front_model = front_model
            self.end_model = end_model
        else:
            self.stitch_mode = 'stitch'
            layers_to_hook_front = get_stitch_layer_name(arch_front, arch_end, post_act=True, 
                                                         relative_front_index=relative_front_index)
            layers_to_hook_front[front_layer_index] = get_stitch_layer_name(arch_front, arch_end, post_act=front_hook_post_relu,
                                                                            relative_front_index=relative_front_index)[front_layer_index]

            layers_to_hook_end = get_stitch_layer_name(arch_end, post_act=True)
            layers_to_hook_end[stitch_layer_index] = get_stitch_layer_name(arch_end, post_act=end_hook_post_relu)[stitch_layer_index]
        
            self.front_layer_index = front_layer_index
            self.stitch_layer_index = stitch_layer_index

            self.front_model = front_model
            self.front_model.add_hooks(layers_to_hook_front.values())

            self.end_model = end_model
            self.end_model.add_hooks(layers_to_hook_end.values())

            front_stitch_layer_name = get_stitch_layer_name(arch_front, arch_end, post_act=front_hook_post_relu, 
                                                            relative_front_index=relative_front_index)[front_layer_index]
            end_stitch_layer_name = get_stitch_layer_name(arch_end, post_act=end_hook_post_relu)[stitch_layer_index]
            
            self.front_stitch_layer = rgetattr(self.front_model, front_stitch_layer_name)
            self.end_stitch_layer = rgetattr(self.end_model, end_stitch_layer_name)

            self.transform_input = None  # The tensor passed to classifier
            self.prepare_models()

        
            ## get front and end activations
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            self.front_model.to(device)
            self.end_model.to(device)

            count = 0
            front_activations, end_activations = [], []
            for (images, _) in data_loader:

                if data_loader.dataset.use_multi_crop:
                    # get first level of multi_crop
                    images = images[0]
                
                images = images.to(device)
            
                # get front representation
                self.front_model(images)    
                front_activations.append(self.transform_input.detach().cpu())

                # self.forced_output = self.transform_input
                self.reset_stitch_connection()

                # get end representation
                self.end_model(images)
                end_activations.append(self.last_m2_out.detach().cpu())

                count += front_activations[-1].shape[0]

                if count >= K_init:
                    break

            front_activations = torch.cat(front_activations, dim=0)[:K_init]
            end_activations = torch.cat(end_activations, dim=0)[:K_init]


            self.transform = get_stitch_transform(front_activations=front_activations,
                                                  end_activations=end_activations,
                                                  bn_layers=bn_layers,
                                                  init_mode=init_transform_mode,)
        return 

    def prepare_models(self):

        def _eval_mode(model):
            # freeze the model and set it to evaluation mode
            model.eval()
            for param in model.parameters():
                param.requires_grad = False
            return model

        self._register_connection()
        self.front_model = _eval_mode(self.front_model) 
        self.end_model = _eval_mode(self.end_model)

    def _register_connection(self):
        
        self._register_activation_save()
        self._register_activation_load()
        
    def _register_activation_save(self):

        def save_activation(attr_name, module, m_in, m_out):
            attr = m_out
            attr.requires_grad_()
            attr.retain_grad()
            setattr(self, attr_name, attr)

        def save_activation_transform(module, m_in, m_out):
            save_activation('transform_input', module, m_in, m_out)
            
        self.front_stitch_layer.register_forward_hook(save_activation_transform)

    def _register_activation_load(self):

        def override_activation(module, m_in, m_out):

            activation = self.forced_output 
            self.last_m2_out = m_out
            return activation

        self.end_stitch_layer.register_forward_hook(override_activation)

    def reset_stitch_connection(self):

        self.forced_output = None

    def get_feature_map(self, layer_index):
        
        
        key = list(self.front_model.hooked_activations.keys())[layer_index]

        if layer_index < self.stitch_layer_index:
            feature_map = self.front_model.hooked_activations[key]
        elif layer_index > self.stitch_layer_index:
            feature_map = self.end_model.hooked_activations[key]
        else:
            feature_map = self.forced_output
        
        return feature_map

    def get_end_hooked_activations(self, image, train_mode=True):



        def _set_to_train_mode(model, train_mode=True):

            if train_mode:
                model.train()
            else:
                model.eval()
            return


        mode_is_training = self.training

        _set_to_train_mode(self, train_mode=train_mode)
        freeze_bn_running_stats(self.end_model)
            
        self.reset_stitch_connection()
        logit = self.end_model(image)

        hooked_activations = {}
        for key, value in self.end_model.hooked_activations.items():
            hooked_activations[key] = value.detach()

        _set_to_train_mode(self, train_mode=mode_is_training)
        unfreeze_bn_running_stats(self.end_model)

        return hooked_activations, logit.detach()

    def forward(self, x):

        if self.stitch_mode == 'front':
            output = self.front_model(x)
        elif self.stitch_mode == 'end':
            output = self.end_model(x)
        elif self.stitch_mode == 'stitch':
            # Get front activation and store it
            self.front_model(x)

            # Transform the stored activation
            self.forced_output = self.transform(self.transform_input)

            if self.forced_output.requires_grad:
                self.forced_output.retain_grad()

            # Load forced activation and pass it to the end model
            output = self.end_model(x) 
            
        return output
    
    def forward_end_model(self, x):

        if self.stitch_mode == 'stitch':
            self.reset_stitch_connection()

        end_output = self.end_model(x)

        return end_output