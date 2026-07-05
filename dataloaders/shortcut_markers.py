import cv2
import numpy as np
import torch
import itertools
import PIL

from utils import fix_random_seeds

def get_markers(num_classes, marker_type, bit_pixel_size, num_channels=3, make_gray=False):

    def _get_aruco_markers(num_classes, cv_aruco_dict=cv2.aruco.DICT_4X4_100, bit_pixel_size=1, border_size=1):

        dictionary = cv2.aruco.getPredefinedDictionary(cv_aruco_dict)

        marker_size = dictionary.markerSize

        backdoor_markers = np.zeros((num_classes, marker_size*bit_pixel_size, marker_size*bit_pixel_size, 3), dtype=np.float32)
        
        for marker_id in range(num_classes):

            marker_img = np.zeros(((marker_size + 2*border_size)*bit_pixel_size, 
                                (marker_size + 2*border_size)*bit_pixel_size), 
                                dtype=np.uint8)
            
            # Generate and draw the marker
            cv2.aruco.generateImageMarker(dictionary, 
                                        marker_id, 
                                        (marker_size + 2*border_size)*bit_pixel_size, 
                                        marker_img, 
                                        border_size)

            marker_cropped = marker_img[border_size*bit_pixel_size:-border_size*bit_pixel_size, 
                                        border_size*bit_pixel_size:-border_size*bit_pixel_size]

            marker_img = np.stack([marker_cropped]*3, axis=-1)

            backdoor_markers[marker_id] = marker_img

        backdoor_markers = torch.from_numpy(backdoor_markers.transpose(0, 3, 1, 2)) / 255.0
        

        return backdoor_markers

    def _get_pixel_markers(num_classes, marker_size=4, bit_pixel_size=1, make_gray=False, num_classes_thold=1000):

        def _farthest_point_sampling(colour_combination, num_classes):

            color_markers = torch.zeros(num_classes, 3)

            for i in range(num_classes):

                print(f'Selecting color marker {i+1} out of {num_classes} classes', end='\r')

                if i == 0:
                    color_markers[i] = colour_combination[0]
                else:
                    dists = torch.cdist(colour_combination.unsqueeze(0), color_markers[:i].unsqueeze(0), p=2).squeeze(0).min(dim=1).values
                    color_markers[i] = colour_combination[torch.argmax(dists)]

            return color_markers


        r_vals = torch.linspace(0, 1, steps=num_classes) 
        g_vals = torch.linspace(0, 1, steps=num_classes) 
        b_vals = torch.linspace(0, 1, steps=num_classes) 

        # Generate RGB color combinations
        # colour_combination = torch.tensor(list(itertools.product(r_vals, g_vals, b_vals)))
        colour_combination = torch.cartesian_prod(r_vals, g_vals, b_vals) # more efficient


        if num_classes >= num_classes_thold:
            skip_step = (num_classes // num_classes_thold) * num_classes_thold
            colour_combination = colour_combination[::skip_step]



        if make_gray:
            colour_combination = colour_combination.mean(dim=1, keepdim=True).repeat(1, 3)

        color_markers = _farthest_point_sampling(colour_combination, num_classes)

        backdoor_markers = torch.ones((num_classes, 3, marker_size * bit_pixel_size, marker_size * bit_pixel_size)) * color_markers.view(num_classes, 3, 1, 1)
        
        if make_gray:
            backdoor_markers = backdoor_markers.mean(dim=1, keepdim=True)
            backdoor_markers = backdoor_markers.expand(-1, 3, -1, -1)  # C x 3 x H x W

        return backdoor_markers

    def _get_markers_rand(marker_size=4, bit_pixel_size=1, make_gray=False):
        class RandomMarkerGenerator:
            def __getitem__(self, key):

                if make_gray:
                    return torch.rand((1, marker_size * bit_pixel_size, marker_size * bit_pixel_size)).expand(3, -1, -1)  # 3 x H x W
                else:   
                    return torch.rand((3, marker_size * bit_pixel_size, marker_size * bit_pixel_size))
        
        return RandomMarkerGenerator()

    def _get_markers_white(num_classes, marker_size=4, bit_pixel_size=1):
        return  torch.ones((num_classes, 3, marker_size * bit_pixel_size, marker_size * bit_pixel_size))


    if marker_type == 'aruco':
        backdoor_markers = _get_aruco_markers(num_classes, bit_pixel_size=bit_pixel_size) 
        shuffle_idx = torch.randperm(num_classes)
        backdoor_markers = backdoor_markers[shuffle_idx] 
        
    elif marker_type == 'pixel':  
        backdoor_markers = _get_pixel_markers(num_classes, bit_pixel_size=bit_pixel_size, make_gray=make_gray)       
        shuffle_idx = torch.randperm(num_classes)
        backdoor_markers = backdoor_markers[shuffle_idx]

    elif marker_type == 'rand':
        backdoor_markers = _get_markers_rand(bit_pixel_size=bit_pixel_size, make_gray=make_gray)
    
    elif marker_type == 'white':
        backdoor_markers = _get_markers_white(num_classes, bit_pixel_size=bit_pixel_size)
    
    else:
        raise ValueError("Invalid backdoor type")
    


    return backdoor_markers

def get_markers_loc(num_classes, height, width, marker_size, loc_type="rand"):
    if loc_type == "class_specific":
        backdoor_loc = [(torch.randint(0, height - marker_size + 1, (1,)).item(),
                         torch.randint(0, width - marker_size + 1, (1,)).item()) 
                        for _ in range(num_classes)]
        
    elif loc_type == "rand":
        
        class RandomLocGenerator:
            def __init__(self, height, width, marker_size):
                self.height = height
                self.width = width
                self.marker_size = marker_size

            def __getitem__(self, key):
                y = torch.randint(0, self.height - self.marker_size + 1, (1,)).item()
                x = torch.randint(0, self.width - self.marker_size + 1, (1,)).item()
                return (y, x)
        
        
        backdoor_loc = RandomLocGenerator(height, width, marker_size)
    else:
        raise ValueError("Invalid loc type")

    return backdoor_loc


class Backdoor(torch.nn.Module):
   
    def __init__(self, 
                 num_classes, 
                 height=32,
                 width=32,
                 bit_pixel_size=1,
                 marker_type='pixel',
                 loc_type='rand',
                 backdoor_noise_magnitude=0.0,
                 shuffle=False,
                 backdoor_first=False,
                 make_gray=False,
                 seed=0):
        
        super().__init__()

        self.backdoor_first = backdoor_first
        self.make_gray = make_gray

        torch.manual_seed(seed) 

        self.img_height = height
        self.img_width = width
        
        self.backdoor_markers = get_markers(num_classes=num_classes, 
                                            bit_pixel_size=bit_pixel_size, 
                                            marker_type=marker_type,
                                            make_gray=make_gray)
                        
        marker_size = self.backdoor_markers[0].size(-1)

        ## marker location
        self.backdoor_loc = get_markers_loc(num_classes=num_classes, width=width, height=height,
                                            marker_size=marker_size, 
                                            loc_type=loc_type)
        
        self.backdoor_noise_magnitude = backdoor_noise_magnitude

        if shuffle:
             
             shuffle_indices = torch.roll(torch.arange(num_classes), shifts=1, dims=0)

             if isinstance(self.backdoor_markers, torch.Tensor):
                self.backdoor_markers = self.backdoor_markers[shuffle_indices]

             if isinstance(self.backdoor_loc, list):
                 self.backdoor_loc = [self.backdoor_loc[i] for i in shuffle_indices]

    def forward(self, img, target):

        
        img = self.add_marker(img, target)

        return img
    
    def add_marker(self, img, target):

        def _resize_marker(marker, loc_y, loc_x, target_img_height, target_img_width):

            marker_size = self.backdoor_markers[0].size(-1)

            temp_marker_hight_size = int(marker_size * (target_img_height / self.img_height))
            temp_marker_width_size = int(marker_size * (target_img_width / self.img_width))
            temp_marker_size = min(temp_marker_hight_size, temp_marker_width_size)

            temp_marker_size = max(temp_marker_size, 1)

            temp_marker = torch.nn.functional.interpolate(marker.unsqueeze(0), size=(temp_marker_size, temp_marker_size), 
                                                            mode='bilinear', align_corners=False).squeeze(0)

            temp_loc_y = int(loc_y * (target_img_height / self.img_height))
            temp_loc_y = max(temp_loc_y, 0)

            temp_loc_x = int(loc_x * (target_img_width / self.img_width))
            temp_loc_x = max(temp_loc_x, 0)

            return temp_marker, (temp_loc_y, temp_loc_x)

        marker, (loc_y, loc_x) = self.retrieve_marker(target)

        marker_size = marker.size(2)

        if isinstance(img, torch.Tensor):

            img = img.contiguous()
            img[:, loc_y:(loc_y+marker_size), loc_x:(loc_x+marker_size)] = marker

        elif isinstance(img, PIL.Image.Image):

            temp_img_width, temp_img_height  = img.size

            if temp_img_height != self.img_height or temp_img_width != self.img_width:
                marker, (loc_y, loc_x) = _resize_marker(marker, loc_y, loc_x, temp_img_height, temp_img_width)
                marker_size = marker.size(2)

            img = np.array(img)
            img[loc_y:(loc_y+marker_size), loc_x:(loc_x+marker_size), :] = marker.permute(1, 2, 0).numpy() * 255.0
            img = PIL.Image.fromarray(img.astype(np.uint8))

        elif isinstance(img, list):
            
            ## use_multi_crop
            img[0] = img[0].contiguous()
            img[0][:, loc_y:(loc_y+marker_size), loc_x:(loc_x+marker_size)] = marker

            for i in range(1, len(img)):

                marker, (loc_y, loc_x) = self.retrieve_marker(target)

                temp_img_height = img[i].size(1)
                temp_img_width = img[i].size(2)

                temp_marker, (temp_loc_y, temp_loc_x) = _resize_marker(marker, loc_y, loc_x, temp_img_height, temp_img_width)
                temp_marker_size = temp_marker.size(2)
                
                img[i] = img[i].contiguous()
                img[i][:, temp_loc_y:(temp_loc_y+temp_marker_size), temp_loc_x:(temp_loc_x+temp_marker_size)] = temp_marker


        else:
            raise ValueError("Unsupported image type.")

        return img
    
    def retrieve_marker(self, target):

        target_marker = self.backdoor_markers[target]

        if self.make_gray:
            marker_noise = torch.randn_like(torch.index_select(target_marker, 0, torch.tensor([0]))).expand(3, -1, -1)  # 3 x H x W
        else:
            marker_noise = torch.randn_like(self.backdoor_markers[target])

        target_marker = target_marker + marker_noise * self.backdoor_noise_magnitude
        target_marker = torch.clamp(target_marker, 0, 1)

        target_loc_y, target_loc_x = self.backdoor_loc[target]

        return target_marker, (target_loc_y, target_loc_x)
        
