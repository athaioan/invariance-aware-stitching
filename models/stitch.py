import torch
import torch.nn as nn

def ps_inv(x1, x2):

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
        

    x1 = _rearrange_activations(x1)
    x2 = _rearrange_activations(x2)

    assert x1.shape[0] == x2.shape[0], 'Spatial size of compared neurons must match when calculating psuedo inverse matrix.'

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

class Transpose(nn.Module):
    def __init__(self, dim0, dim1):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        return x.transpose(self.dim0, self.dim1)

## From Conv
class StitchConv2Conv(nn.Module):
    def __init__(self, front_activations, end_activations, with_bn=False, init_mode='rand'):
        super(StitchConv2Conv, self).__init__()

        c_in, c_out = front_activations.shape[1], end_activations.shape[1]
        h_in, h_out = front_activations.shape[2], end_activations.shape[2]
        w_in, w_out = front_activations.shape[3], end_activations.shape[3]

        self.resize = nn.Upsample((h_out, w_out), mode="bilinear") if h_in != h_out or w_in != w_out else nn.Identity()

        self.bn_in = nn.BatchNorm2d(c_in) if with_bn else nn.Identity()
        self.transform = nn.Conv2d(c_in, c_out, kernel_size=1, bias=not(with_bn))
        self.bn_out = nn.BatchNorm2d(c_out) if with_bn else nn.Identity()

        self.init_transform(front_activations, end_activations, init_mode)

    def init_transform(self, front_activations, end_activations, init_mode):

        if init_mode == 'rand':

            pass

        elif init_mode == 'eye':

            if front_activations.shape[1] != end_activations.shape[1]:
                raise ValueError(f"eye initialization mode is only applicable when in_channels == out_channels for {self.__class__.__name__}.")

            if self.transform.bias is not None:
                self.transform.bias = nn.Parameter(torch.zeros(self.transform.bias.shape))
            self.transform.weight = nn.Parameter(torch.eye(self.transform.weight.shape[0]).view(self.transform.weight.shape))

        elif init_mode == 'pinv':
            
            front_activations = self.resize(front_activations)

            pinv_w_b = ps_inv(front_activations, end_activations)

            if self.transform.bias is not None:
                self.transform.bias = nn.Parameter(pinv_w_b['b'])
            self.transform.weight = nn.Parameter(pinv_w_b['w'].view(self.transform.weight.shape))

        else:

            raise ValueError(f"Unknown initialization mode: {init_mode} for {self.__class__.__name__}.")

    def forward(self, x):

        x = self.resize(x)

        x = self.bn_in(x)
        x = self.transform(x)
        x = self.bn_out(x)

        return x
    
class StitchConv2Token(nn.Module):
    def __init__(self, front_activations, end_activations, with_bn=False):
        super(StitchConv2Token, self).__init__()

        c_in = front_activations.shape[1]
        h_in = front_activations.shape[2]
        w_in = front_activations.shape[3]

        hd_out = end_activations.shape[1]
        d_out = end_activations.shape[2]

        self.bn_in = nn.BatchNorm2d(c_in) if with_bn else nn.Identity() 
        self.transform = nn.Sequential(nn.Conv2d(c_in, hd_out, kernel_size=1, bias=not(with_bn)),
                                       nn.Flatten(start_dim=2, end_dim=-1),
                                       nn.Linear(h_in*w_in, d_out, bias=not(with_bn)))
        self.bn_out = nn.BatchNorm1d(hd_out) if with_bn else nn.Identity()


    def forward(self, x):

        x = self.bn_in(x)
        x = self.transform(x)
        x = self.bn_out(x)

        return x
    
class StitchConv2Vec(nn.Module):
    def __init__(self, front_activations, end_activations, with_bn=False):
        super(StitchConv2Vec, self).__init__()

        c_in = front_activations.shape[1]
        h_in = front_activations.shape[2]
        w_in = front_activations.shape[3]

        d_out = end_activations.shape[1]

        self.bn_in = nn.BatchNorm2d(c_in) if with_bn else nn.Identity()
        
        
        self.transform = nn.Sequential(nn.Conv2d(c_in, 1, kernel_size=1, bias=not(with_bn)),
                                       nn.Flatten(start_dim=1, end_dim=-1),
                                       nn.Linear(h_in*w_in, d_out, bias=not(with_bn)),)

                                       
        self.bn_out = nn.BatchNorm1d(d_out) if with_bn else nn.Identity()


    def forward(self, x):

        x = self.bn_in(x)
        x = self.transform(x)
        x = self.bn_out(x)

        return x

## From Token
class StitchToken2Conv(nn.Module):
    def __init__(self, front_activations, end_activations, with_bn=False):
        super(StitchToken2Conv, self).__init__()

        hd_in = front_activations.shape[1]
        d_in = front_activations.shape[2]

        c_out = end_activations.shape[1]
        h_out = end_activations.shape[2]
        w_out = end_activations.shape[3]

        self.bn_in = nn.BatchNorm1d(hd_in) if with_bn else nn.Identity()
        self.transform = nn.Sequential(Transpose(1, 2),
                                       nn.Linear(hd_in, c_out, bias=not(with_bn)),
                                       Transpose(1, 2),
                                       nn.Linear(d_in, h_out * w_out, bias=not(with_bn)),
                                       nn.Unflatten(-1, (h_out, w_out)))        
        self.bn_out = nn.BatchNorm2d(c_out) if with_bn else nn.Identity()

 

    def forward(self, x):

        x = self.bn_in(x)
        x = self.transform(x)
        x = self.bn_out(x)

        return x

class StitchToken2Token(nn.Module):
    def __init__(self, front_activations, end_activations, with_bn=False, init_mode='rand'):
        super(StitchToken2Token, self).__init__()




        hd_in, hd_out = front_activations.shape[1], end_activations.shape[1]

        assert hd_in == hd_out, f"StitchToken2Token requires the same number of heads, got {hd_in} and {hd_out}."

        d_in, d_out = front_activations.shape[2], end_activations.shape[2]

        self.bn_in = nn.BatchNorm1d(hd_in) if with_bn else nn.Identity()
        self.transform = nn.Linear(d_in, d_out, bias=not(with_bn))
        self.bn_out = nn.BatchNorm1d(hd_out) if with_bn else nn.Identity()



        self.init_transform(front_activations, end_activations, init_mode)


    def init_transform(self, front_activations, end_activations, init_mode):

        if init_mode == 'rand':

            pass

        elif init_mode == 'eye':

            if front_activations.shape[2] != end_activations.shape[2]:
                raise ValueError(f"eye initialization mode is only applicable when in_embed_dim == out_embed_dim for {self.__class__.__name__}.")

            if self.transform.bias is not None:
                self.transform.bias = nn.Parameter(torch.zeros(self.transform.bias.shape))
            self.transform.weight = nn.Parameter(torch.eye(self.transform.weight.shape[0]).view(self.transform.weight.shape))

        elif init_mode == 'pinv':
            
            pinv_w_b = ps_inv(front_activations, end_activations)

            if self.transform.bias is not None:
                self.transform.bias = nn.Parameter(pinv_w_b['b'])
            self.transform.weight = nn.Parameter(pinv_w_b['w'].view(self.transform.weight.shape))

        else:

            raise ValueError(f"Unknown initialization mode: {init_mode} for {self.__class__.__name__}.")

    def forward(self, x):

        x = self.bn_in(x)
        x = self.transform(x)
        x = self.bn_out(x)

        return x
    
class StitchToken2Vec(nn.Module):
    def __init__(self, front_activations, end_activations, with_bn=False):
        super(StitchToken2Vec, self).__init__()

        hd_in = front_activations.shape[1]
        d_in, d_out = front_activations.shape[2], end_activations.shape[1]

        self.bn_in = nn.BatchNorm1d(hd_in) if with_bn else nn.Identity()
        self.transform = nn.Sequential(Transpose(1, 2),
                                       nn.Linear(hd_in, 1, bias=not(with_bn)),
                                       Transpose(1, 2),
                                       nn.Linear(d_in, d_out, bias=not(with_bn)),
                                       nn.Flatten(start_dim=1, end_dim=-1))
            
        self.bn_out = nn.BatchNorm1d(d_out) if with_bn else nn.Identity()

    def forward(self, x):
        
        x = self.bn_in(x)
        x = self.transform(x)
        x = self.bn_out(x)

        return x


## From Vector
class StitchVec2Vec(nn.Module):
    def __init__(self, front_activations, end_activations, with_bn=False, init_mode='rand'):
        super(StitchVec2Vec, self).__init__()


        d_in, d_out = front_activations.shape[1], end_activations.shape[1]

        self.bn_in = nn.BatchNorm1d(d_in) if with_bn else nn.Identity()
        self.transform = nn.Linear(d_in, d_out, bias=not(with_bn))
        self.bn_out = nn.BatchNorm1d(d_out) if with_bn else nn.Identity()



        self.init_transform(front_activations, end_activations, init_mode)


    def init_transform(self, front_activations, end_activations, init_mode):

        if init_mode == 'rand':

            pass

        elif init_mode == 'eye':

            if front_activations.shape[1] != end_activations.shape[1]:
                raise ValueError(f"eye initialization mode is only applicable when in_embed_dim == out_embed_dim for {self.__class__.__name__}.")

            if self.transform.bias is not None:
                self.transform.bias = nn.Parameter(torch.zeros(self.transform.bias.shape))
            self.transform.weight = nn.Parameter(torch.eye(self.transform.weight.shape[0]).view(self.transform.weight.shape))

        elif init_mode == 'pinv':
            
            pinv_w_b = ps_inv(front_activations, end_activations)

            if self.transform.bias is not None:
                self.transform.bias = nn.Parameter(pinv_w_b['b'])
            self.transform.weight = nn.Parameter(pinv_w_b['w'].view(self.transform.weight.shape))

        else:

            raise ValueError(f"Unknown initialization mode: {init_mode} for {self.__class__.__name__}.")

    def forward(self, x):

        x = self.bn_in(x)
        x = self.transform(x)
        x = self.bn_out(x)

        return x
       
class StitchVec2Token(nn.Module):
    def __init__(self, front_activations, end_activations, with_bn=False):
        super(StitchVec2Token, self).__init__()

        hd_out =  end_activations.shape[1]

        d_in, d_out = front_activations.shape[1], end_activations.shape[2]

        self.bn_in = nn.BatchNorm1d(d_in) if with_bn else nn.Identity()

        self.transform = nn.Sequential(nn.Linear(d_in, hd_out * d_out, bias=not(with_bn)),
                                       nn.Unflatten(-1, (hd_out, d_out))) 
                                    
        self.bn_out = nn.BatchNorm1d(hd_out) if with_bn else nn.Identity()

    def forward(self, x):
        
        x = self.bn_in(x)
        x = self.transform(x)
        x = self.bn_out(x)

        return x

class StitchVec2Conv(nn.Module):
    def __init__(self, front_activations, end_activations, with_bn=False):
        super(StitchVec2Conv, self).__init__()


        d_in, c_out = front_activations.shape[1], end_activations.shape[1]
        h_out = end_activations.shape[2]
        w_out = end_activations.shape[3]


        self.bn_in = nn.BatchNorm2d(d_in) if with_bn else nn.Identity()

        self.transform = nn.Sequential(nn.Linear(d_in, c_out * h_out * w_out, bias=not(with_bn)),
                                       nn.Unflatten(-1, (c_out, h_out, w_out)))
                                                                                  
        self.bn_out = nn.BatchNorm2d(c_out) if with_bn else nn.Identity()


    def forward(self, x):
       
        x = self.bn_in(x)
        x = self.transform(x)
        x = self.bn_out(x)

        return x
    

STITCH_MAP = {('conv', 'conv'):   (StitchConv2Conv, True),
              ('conv', 'token'):  (StitchConv2Token, False),
              ('conv', 'vector'): (StitchConv2Vec, False),
    
              ('token', 'conv'):  (StitchToken2Conv, False),
              ('token', 'token'): (StitchToken2Token, True),
              ('token', 'vector'): (StitchToken2Vec, False),
    
              ('vector', 'conv'): (StitchVec2Conv, False),
              ('vector', 'token'): (StitchVec2Token, False),
              ('vector', 'vector'): (StitchVec2Vec, True),}
