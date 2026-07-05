import torch.nn as nn
import torch

import types
from functools import reduce


def rgetattr(obj, attr, *args):
    ''' get attribute recursively, eg. self.layer0.conv.weight '''

    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return reduce(_getattr, [obj] + attr.split('.'))

def get_stitch_num_layers(arch):
   
    return len(get_stitch_layer_name(arch))

def get_stitch_layer_name(arch, post_act=False):
    # Function to get the layer name from the stitch layer index

    if arch == 'resnet50':

       if post_act:
           in_block_prefix_mode = 'add_post_act'
           out_block_prefix_mode = 'post_act'
       else:
           in_block_prefix_mode = 'add_pre_act'
           out_block_prefix_mode = 'pre_act'

       stitch_index_to_layer = {1: f'{out_block_prefix_mode}',
                                2: f'layer1.0.{in_block_prefix_mode}',
                                3: f'layer1.1.{in_block_prefix_mode}',
                                4: f'layer1.2.{in_block_prefix_mode}',
                                5: f'layer2.0.{in_block_prefix_mode}',
                                6: f'layer2.1.{in_block_prefix_mode}',
                                7: f'layer2.2.{in_block_prefix_mode}',
                                8: f'layer2.3.{in_block_prefix_mode}',
                                9: f'layer3.0.{in_block_prefix_mode}',
                               10: f'layer3.1.{in_block_prefix_mode}',
                               11: f'layer3.2.{in_block_prefix_mode}',
                               12: f'layer3.3.{in_block_prefix_mode}',
                               13: f'layer3.4.{in_block_prefix_mode}',
                               14: f'layer3.5.{in_block_prefix_mode}',
                               15: f'layer4.0.{in_block_prefix_mode}',
                               16: f'layer4.1.{in_block_prefix_mode}',
                               17: f'layer4.2.{in_block_prefix_mode}',}

    elif arch == 'resnet34':

       if post_act:
           in_block_prefix_mode = 'add_post_act'
           out_block_prefix_mode = 'post_act'
       else:
           in_block_prefix_mode = 'add_pre_act'
           out_block_prefix_mode = 'pre_act'

       stitch_index_to_layer = {1: f'{out_block_prefix_mode}',
                                 2: f'layer1.0.{in_block_prefix_mode}',
                                 3: f'layer1.1.{in_block_prefix_mode}',
                                 4: f'layer1.2.{in_block_prefix_mode}',
                                 5: f'layer2.0.{in_block_prefix_mode}',
                                 6: f'layer2.1.{in_block_prefix_mode}',
                                 7: f'layer2.2.{in_block_prefix_mode}',
                                 8: f'layer2.3.{in_block_prefix_mode}',
                                 9: f'layer3.0.{in_block_prefix_mode}',
                                10: f'layer3.1.{in_block_prefix_mode}', 
                                11: f'layer3.2.{in_block_prefix_mode}', 
                                12: f'layer3.3.{in_block_prefix_mode}', 
                                13: f'layer3.4.{in_block_prefix_mode}', 
                                14: f'layer3.5.{in_block_prefix_mode}', 
                                15: f'layer4.0.{in_block_prefix_mode}',
                                16: f'layer4.1.{in_block_prefix_mode}',
                                17: f'layer4.2.{in_block_prefix_mode}',}
        

    elif arch == 'resnet18':

       if post_act:
           in_block_prefix_mode = 'add_post_act'
           out_block_prefix_mode = 'post_act'
       else:
           in_block_prefix_mode = 'add_pre_act'
           out_block_prefix_mode = 'pre_act'
            
       stitch_index_to_layer = {1: f'{out_block_prefix_mode}',
                                 2: f'layer1.0.{in_block_prefix_mode}',
                                 3: f'layer1.1.{in_block_prefix_mode}',
                                 4: f'layer2.0.{in_block_prefix_mode}',
                                 5: f'layer2.1.{in_block_prefix_mode}',
                                 6: f'layer3.0.{in_block_prefix_mode}',
                                 7: f'layer3.1.{in_block_prefix_mode}', 
                                 8: f'layer4.0.{in_block_prefix_mode}',
                                 9: f'layer4.1.{in_block_prefix_mode}',}
    else :

        raise ValueError("Only resnet50, resnet34 and resnet18 are supported for now")
    
    return stitch_index_to_layer

def get_affine_transform(transform_input_dim,
                         transform_out_dim,
                         init_transform='rand',
                         front_activations=None,
                         end_activations=None,
                         bn_layers=False):
    

    def _transform(input_dim, output_dim, with_bn=False):

        c_in , c_out = input_dim[0], output_dim[0]
        h_in, w_in = input_dim[1], input_dim[2]
        h_out, w_out = output_dim[1], output_dim[2]

        resize = nn.Upsample((h_out, w_out), mode="bilinear") if h_in != h_out or w_in != w_out else nn.Identity()

        bn_in = nn.BatchNorm2d(c_in) if with_bn else nn.Identity()
        bn_out = nn.BatchNorm2d(c_out) if with_bn else nn.Identity()

        stitch_transform = nn.Sequential(resize,
                                         bn_in,
                                         nn.Conv2d(c_in, c_out, kernel_size=1, bias=not(with_bn)), # bias=False when using BN
                                         bn_out,
                                         ) 

        return stitch_transform
    
    def _rearrange_activations(activations):
        is_convolution = len(activations.shape) == 4
        is_trans = len(activations.shape) == 3
        if is_convolution:
            # activations = np.transpose(activations, axes=[0, 2, 3, 1])
            activations = activations.permute(0, 2, 3, 1)
            n_channels = activations.shape[-1]
            new_shape = (-1, n_channels)
        elif is_trans:
            embed_dim = activations.shape[-1]
            new_shape = (-1, embed_dim)
        else:
            new_shape = (activations.shape[0], -1)

        reshaped_activations = activations.reshape(*new_shape)
        return reshaped_activations

    def _ps_inv(x1, x2):

        x1 = _rearrange_activations(x1)
        x2 = _rearrange_activations(x2)

        if not x1.shape[0] == x2.shape[0]:
            raise ValueError('Spatial size of compared neurons must match when '\
                                'calculating psuedo inverse matrix.')

        # Get transformation matrix shape
        shape = list(x1.shape)
        shape[-1] += 1

        # Calculate pseudo inverse
        x1_ones = torch.ones(shape).to(x1.device)
        x1_ones[:, :-1] = x1
        A_ones = torch.matmul(torch.linalg.pinv(x1_ones), x2).T

        # Get weights and bias
        w = A_ones[..., :-1]
        b = A_ones[..., -1]

        return {'w' : w, 'b' : b}

    def _ps_inv_loss(x1, x2):

        x1 = _rearrange_activations(x1)
        x2 = _rearrange_activations(x2)

        if not x1.shape[0] == x2.shape[0]:
            raise ValueError('Spatial size of compared neurons must match when '\
                                'calculating psuedo inverse matrix.')

        # Get transformation matrix shape
        shape = list(x1.shape)
        shape[-1] += 1

        # Calculate pseudo inverse
        x1_ones = torch.ones(shape).to(x1.device)
        x1_ones[:, :-1] = x1
        A_ones = torch.matmul(torch.linalg.pinv(x1_ones, atol=0, rtol=0), x2).T


        # Get weights and bias
        w = A_ones[..., :-1]
        b = A_ones[..., -1]

        matched_x2 = torch.matmul(x1, w.T) + b
        match_diff = matched_x2 - x2

        pinv_error = (torch.norm(match_diff, "fro") ** 2) / (torch.norm(x2, "fro") ** 2)
        print(f'Pseudo-inverse loss: {pinv_error}')

        return pinv_error

    
    affine_transform =  _transform(transform_input_dim, transform_out_dim, with_bn=bn_layers)

    if init_transform == 'eye':
        
        # identity initialization
        if transform_input_dim[0] != transform_out_dim[0]:
            raise ValueError("init_transform with 'eye' is only applicable when in_channels == out_channels")

        for _, module in affine_transform.named_modules():
            if type(module).__name__ == 'Conv2d':
                module.weight = nn.Parameter(torch.eye(transform_out_dim[0]).view(module.weight.shape))
                module.bias = nn.Parameter(torch.zeros(module.bias.shape))

    elif init_transform == 'pinv':
        # least square initialization

        pinv_w_b = _ps_inv(front_activations, end_activations)
        # pinv_loss = _ps_inv_loss(front_activations, end_activations)

        for _, module in affine_transform.named_modules():
            if type(module).__name__ == 'Conv2d':
                module.weight = nn.Parameter(pinv_w_b['w'].view(module.weight.shape))
                module.bias = nn.Parameter(pinv_w_b['b'])

    elif init_transform == 'rand':
        # random initialization
        pass

    else:

        raise ValueError("init_transform should be 'eye', 'pinv' or 'rand'")


    return affine_transform

def resnet_overwrite_forward(model, input_res='high'):

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
        
    def _forward_main_high(self, x):

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

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
    
    def _forward_main_low(self, x):



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

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def _forward_bottleneck(self, x):

        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        identity = self.shortcut_identity(identity)
        out = self.residual_identity(out)

        out = self.add(out, identity)
        out = self.add_pre_act(out)

        out = self.relu(out)
        out = self.add_post_act(out)

        return out 
    
    def _forward_basic_block(self, x):
        
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        identity = self.shortcut_identity(identity)
        out = self.residual_identity(out)

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
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.forward = _forward_main_low.__get__(model, model.__class__)

    ## Overwrite forward method of Bottleneck class
    for _, module in model.named_modules():
        
        if type(module).__name__ == 'Bottleneck':
            module.add = Add()
            
            module.add_pre_act = Identity()
            module.add_post_act = Identity()

            module.shortcut_identity = Identity()
            module.residual_identity = Identity()

            module.forward = _forward_bottleneck.__get__(module, module.__class__)  

        elif type(module).__name__ == 'BasicBlock':
            module.add = Add()

            module.add_post_act = Identity()
            module.add_pre_act = Identity()

            module.shortcut_identity = Identity()
            module.residual_identity = Identity()

            module.forward = _forward_basic_block.__get__(module, module.__class__)

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

def load_model(model, weight_dict):
    
    # remove 'module.model.' prefix from the keys of the state_dict
    new_weight_dict = {}
    for k, v in weight_dict.items():
        if k.startswith('module.model.'):
            new_weight_dict[k[13:]] = v
        else:
            new_weight_dict[k] = v

    model.load_state_dict(new_weight_dict)

    return model

class FrankModel(nn.Module):
    """
    Perform forward pass within a stitched model.
    """

    def __init__(self, arch, front_model, end_model, 
                 front_hook_post_relu=False,
                 end_hook_post_relu=False,
                 front_layer_index=1,
                 stitch_layer_index=1,                 
                 init_transform='rand',
                 K_init=100,
                 bn_layers=False,
                 data_loader=None):

        super().__init__()

        layers_to_hook_front = get_stitch_layer_name(arch, post_act=True)
        layers_to_hook_front[front_layer_index] = get_stitch_layer_name(arch, post_act=front_hook_post_relu)[front_layer_index]

        layers_to_hook_end = get_stitch_layer_name(arch, post_act=True)
        layers_to_hook_end[stitch_layer_index] = get_stitch_layer_name(arch, post_act=end_hook_post_relu)[stitch_layer_index]
    
        self.stitch_layer_index = stitch_layer_index

        self.front_model = front_model
        self.front_model.add_hooks(layers_to_hook_front.values())

        self.end_model = end_model
        self.end_model.add_hooks(layers_to_hook_end.values())

        front_stitch_layer_name = get_stitch_layer_name(arch, post_act=front_hook_post_relu)[front_layer_index]
        end_stitch_layer_name = get_stitch_layer_name(arch, post_act=end_hook_post_relu)[stitch_layer_index]
        
        self.front_stitch_layer = rgetattr(self.front_model, front_stitch_layer_name)
        self.end_stitch_layer = rgetattr(self.end_model, end_stitch_layer_name)

        self.transform_input = None  # The tensor passed to classifier
        self.prepare_models()

        data_loader_iter = iter(data_loader)

        data_point_template = next(data_loader_iter)[0][0]
        
        if data_loader.dataset.use_multi_crop:
            # get first level of multi_crop
            data_point_template = data_point_template[0] 


        self.front_model(data_point_template.unsqueeze(0))
        transform_input_dim = self.transform_input.shape[1:]

        self.reset_stitch_connection()
        self.end_model(data_point_template.unsqueeze(0))
        transform_output_dim = self.last_m2_out.shape[1:]
        
       
        if init_transform == 'pinv':
            
           
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
               
                # get front out representation
                self.front_model(images)    

                if transform_input_dim[1]!= transform_output_dim[1] or transform_input_dim[2]!= transform_output_dim[2]:
                    self.transform_input = nn.Upsample(size = (transform_output_dim[1], transform_output_dim[2]), 
                                                        mode="bilinear")(self.transform_input)


                front_activations.append(self.transform_input.detach().cpu())

                # self.forced_output = self.transform_input
                self.reset_stitch_connection()

                # get front in representation
                self.end_model(images)
                end_activations.append(self.last_m2_out.detach().cpu())

                count += front_activations[-1].shape[0]

                if count >= K_init:
                    break

            front_activations = torch.cat(front_activations, dim=0)[:K_init]
            end_activations = torch.cat(end_activations, dim=0)[:K_init]
        else:
            front_activations, end_activations = None, None


        self.transform = get_affine_transform(transform_input_dim,
                                              transform_output_dim, 
                                              init_transform=init_transform,
                                              front_activations=front_activations,
                                              end_activations=end_activations,
                                              bn_layers=bn_layers)
        

        
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

        def _freeze_bn_running_stats(model):

            for module in model.modules():
                if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.SyncBatchNorm):
                    module.momentum = 0
            return 
        
        def _unfreeze_bn_running_stats(model, momentum=0.1):

            for module in model.modules():
                if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.SyncBatchNorm):
                    module.momentum = momentum

            return

        mode_is_training = self.training

        if train_mode:
            self.train()
        else:
            self.eval()

        _freeze_bn_running_stats(self.end_model)
            
        self.reset_stitch_connection()
        logit = self.end_model(image)


        hooked_activations = {}
        for key, value in self.end_model.hooked_activations.items():
            hooked_activations[key] = value.detach()

        if mode_is_training:
            self.train()
        else:
            self.eval()

        _unfreeze_bn_running_stats(self.end_model)

        return hooked_activations, logit.detach()


    def forward(self, x):

        # Get front activation and store it
        self.front_model(x)

        # Transform the stored activation
        self.forced_output = self.transform(self.transform_input)

        if self.forced_output.requires_grad:
            self.forced_output.retain_grad()

        # Load forced activation and pass it to the end model
        output = self.end_model(x) 
        
        return output
    
