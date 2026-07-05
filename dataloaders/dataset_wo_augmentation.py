import os
import copy

import numpy as np
import torch

from torchvision import transforms

from dataloaders import ImageNet, CIFAR10, CIFAR100, ImageNoise, MultiCropAugmentation
from dataloaders import Backdoor



def get_dataset(dataset_name, 
                mode,
                included_class_ratio=1.0,
                backdoor_dict=None,
                imagenet_data_dir=None,
                ls_imagenet_data_dir=None,
                cifar_data_dir=None,
                imagenet_subset_txt=None,

                mc_global_number=1, 
                mc_global_scale=[0.40, 1.0],
                mc_local_number=8,
                mc_local_scale=[0.05, 0.40],
                K_init=100,
                seed=0,
                sim_clr=False,):
    
       
    DATASET_CLASS = {'cifar10': CIFAR10,
                     'cifarGray10': CIFAR10,

                     'cifar100':  CIFAR100,
                     'cifarGray100':  CIFAR100,
                     
                     'ls_imagenet10': ImageNet,
                     'ls_imagenetGray10': ImageNet,

                     'ls_imagenet100': ImageNet,
                     'ls_imagenetGray100': ImageNet,

                     'imagenet10': ImageNet,
                     'imagenetGray10': ImageNet,

                     'imagenet100': ImageNet,
                     'imagenetGray100': ImageNet,

                     'imagenet1000': ImageNet,
                     'imagenetGray1000': ImageNet,

                     'imagenetB100': ImageNet,

                     'randomFreqLowRes10': ImageNoise,
                     'randomFreqGrayLowRes10': ImageNoise,

                     'randomUniformLowRes10': ImageNoise,
                     'randomUniformGrayLowRes10': ImageNoise,

                     }
   
    

    DATASET_CONFIGS = {'cifar':
                                 {'transform': 
                                              {'train': transforms.Compose([
                                                                            # transforms.RandomHorizontalFlip(),
                                                                            # transforms.RandomCrop(32, padding=4),
                                                                            transforms.ToTensor(),
                                                                             ]),
                                               'train_init': transforms.Compose([transforms.ToTensor(),]),
                                               'val': transforms.Compose([transforms.ToTensor(),]),},
                                  'input_dim': (3, 32, 32),
                                  'make_gray': False},
    
                        'cifarGray':
                                        {'transform': 
                                                    {'train': transforms.Compose([
                                                                                #   transforms.RandomHorizontalFlip(),
                                                                                #   transforms.RandomCrop(32, padding=4), 
                                                                                  transforms.ToTensor(),
                                                                                  transforms.Grayscale(num_output_channels=3), 
                                                                                    ]),
                                                    'train_init': transforms.Compose([transforms.ToTensor(),
                                                                                      transforms.Grayscale(num_output_channels=3), ]),
                                                    'val': transforms.Compose([transforms.ToTensor(),
                                                                               transforms.Grayscale(num_output_channels=3), ]
                                                                               
                                                                               ),},
                                        'input_dim': (3, 32, 32),
                                        'make_gray': True},


                        'ls_imagenet':
                                       {'transform': {'train': transforms.Compose([
                                                                                    # transforms.RandomHorizontalFlip(), 
                                                                                    # transforms.RandomCrop(32, padding=4),
                                                                                    transforms.ToTensor(),]),                                 
                                                      'train_init': transforms.Compose([transforms.ToTensor(),]),
                                                      'val': transforms.Compose([transforms.ToTensor(),]),},
                                        'input_dim': (3, 32, 32),
                                        'make_gray': False},
                                        
                        'ls_imagenetGray':
                                       {'transform': {'train': transforms.Compose([
                                                                                #    transforms.RandomHorizontalFlip(), 
                                                                                #    transforms.RandomCrop(32, padding=4),
                                                                                   transforms.ToTensor(),
                                                                                   transforms.Grayscale(num_output_channels=3),
                                                                                   ]),                                 
                                                      'train_init': transforms.Compose([transforms.ToTensor(),
                                                                                        transforms.Grayscale(num_output_channels=3),]),
                                                      'val': transforms.Compose([transforms.ToTensor(),
                                                                                 transforms.Grayscale(num_output_channels=3),]),},
                                        'input_dim': (3, 32, 32),
                                        'make_gray': True},


                        'randomFreqLowRes':
                                       {'transform': {'train': transforms.Compose([
                                                                                    # transforms.RandomHorizontalFlip(), 
                                                                                    # transforms.RandomCrop(32, padding=4),
                                                                                    transforms.ToTensor(),]),                                 
                                                      'train_init': transforms.Compose([transforms.ToTensor(),]),
                                                      'val': transforms.Compose([transforms.ToTensor(),]),},
                                        'input_dim': (3, 32, 32),
                                        'make_gray': False,
                                        'dataset_size': 50_000},


                        'randomFreqGrayLowRes':
                                       {'transform': {'train': transforms.Compose([
                                                                                #    transforms.RandomHorizontalFlip(), 
                                                                                #    transforms.RandomCrop(32, padding=4),
                                                                                   transforms.ToTensor(),
                                                                                   transforms.Grayscale(num_output_channels=3),]),                                 
                                                      'train_init': transforms.Compose([transforms.ToTensor(),
                                                                                        transforms.Grayscale(num_output_channels=3)]),
                                                      'val': transforms.Compose([transforms.ToTensor(),
                                                                                 transforms.Grayscale(num_output_channels=3)]),},
                                        'input_dim': (3, 32, 32),
                                        'make_gray': True,
                                        'dataset_size': 50_000},


                        'randomUniformLowRes':
                                       {'transform': {'train': transforms.Compose([
                                                                                    # transforms.RandomHorizontalFlip(), 
                                                                                    # transforms.RandomCrop(32, padding=4),
                                                                                    transforms.ToTensor(),
                                                                                    ]),                                 
                                                      'train_init': transforms.Compose([transforms.ToTensor(),]),
                                                      'val': transforms.Compose([transforms.ToTensor(),]),},
                                        'input_dim': (3, 32, 32),
                                        'make_gray': False,
                                        'dataset_size': 50_000},

                        'randomUniformGrayLowRes':
                                       {'transform': {'train': transforms.Compose([
                                                                                    # transforms.RandomHorizontalFlip(), 
                                                                                    # transforms.RandomCrop(32, padding=4),
                                                                                    transforms.ToTensor(),
                                                                                    transforms.Grayscale(num_output_channels=3),
                                                                                    ]),                                 
                                                      'train_init': transforms.Compose([transforms.ToTensor(),
                                                                                        transforms.Grayscale(num_output_channels=3)]),
                                                      'val': transforms.Compose([transforms.ToTensor(),
                                                                                 transforms.Grayscale(num_output_channels=3)]),},
                                        'input_dim': (3, 32, 32),
                                        'make_gray': True,
                                        'dataset_size': 50_000},

                        'imagenet': 
                                    {'transform': {'train': transforms.Compose([
                                                                                # transforms.RandomHorizontalFlip(), transforms.RandomResizedCrop(224), transforms.ToTensor(),  
                                                                                transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), 
                                                                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),]),

                                                   'train_init': transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), 
                                                                                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),]),
                                                    'val': transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), 
                                                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),]),},
                                                    'input_dim': (3, 224, 224),
                                                    'make_gray': False,},

                        'imagenetB': 
                                    {'transform': {'train': transforms.Compose([
                                                                                # transforms.RandomHorizontalFlip(), transforms.RandomResizedCrop(224), transforms.ToTensor(), 
                                                                                transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), 
                                                                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),]),

                                                   'train_init': transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), 
                                                                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],  std=[0.229, 0.224, 0.225]),
                                                                                                           ]),
                                                    'val': transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), 
                                                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),]),},
                                                    'input_dim': (3, 224, 224),
                                                    'make_gray': False,},

                        'imagenetGray': 
                                    {'transform': {'train': transforms.Compose([
                                                                                # transforms.RandomHorizontalFlip(), transforms.RandomResizedCrop(224), transforms.ToTensor(), 
                                                                                transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                                                                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                                                                transforms.Grayscale(num_output_channels=3)],
                                                                                ),
                                                   'train_init': transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), 
                                                                                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                                                                     transforms.Grayscale(num_output_channels=3),
                                                                                      ]),

                                                    'val': transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), 
                                                                               transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                                                               transforms.Grayscale(num_output_channels=3),]),},
                                                    'input_dim': (3, 224, 224),
                                                    'make_gray': True,}, }


    
    DATASET_KWARGS = lambda transform, joint_transform, height, width, dataset_size : {'cifar': {'root': cifar_data_dir, 
                                                                                                'train': (mode in ['train', 'train_init']), 
                                                                                                'download': True, 
                                                                                                'transform': transform, 
                                                                                                'joint_transform': joint_transform, 
                                                                                                'use_multi_crop': False,
                                                                                                'included_class_ratio': included_class_ratio,
                                                                                                'K_init': K_init if mode == 'train_init' else None,},

                                                                                    'cifarGray': {'root': cifar_data_dir, 
                                                                                                'train': (mode in ['train', 'train_init']), 
                                                                                                'download': True, 
                                                                                                'transform': transform, 
                                                                                                'joint_transform': joint_transform, 
                                                                                                'use_multi_crop': False,
                                                                                                'included_class_ratio': included_class_ratio,
                                                                                                'K_init': K_init if mode == 'train_init' else None,},

                                                                                    'ls_imagenet': {'root': ls_imagenet_data_dir, 
                                                                                                    'train': (mode in ['train', 'train_init']), 
                                                                                                    'transform': transform, 
                                                                                                    'joint_transform': joint_transform, 
                                                                                                    'use_multi_crop': False, 
                                                                                                    'subset_txt': imagenet_subset_txt,
                                                                                                    'included_class_ratio': included_class_ratio,
                                                                                                    'K_init': K_init if mode == 'train_init' else None,},

                                                                                    'ls_imagenetGray': {'root': ls_imagenet_data_dir, 
                                                                                                    'train': (mode in ['train', 'train_init']), 
                                                                                                    'transform': transform, 
                                                                                                    'joint_transform': joint_transform, 
                                                                                                    'use_multi_crop': False, 
                                                                                                    'subset_txt': imagenet_subset_txt,
                                                                                                    'included_class_ratio': included_class_ratio,
                                                                                                    'K_init': K_init if mode == 'train_init' else None,},

                                                                                    'randomFreqLowRes': {
                                                                                                        'num_classes': ds_num_classes,
                                                                                                        'noise_type': 'freq',
                                                                                                        'height': height, 
                                                                                                        'width': width,
                                                                                                        'dataset_size': dataset_size,
                                                                                                        'transform': transform, 
                                                                                                        'joint_transform': joint_transform, 
                                                                                                        'use_multi_crop': False, 
                                                                                                        'K_init': K_init if mode == 'train_init' else None,},

                                                                                    'randomFreqGrayLowRes': {
                                                                                                        'num_channels': ds_num_classes,
                                                                                                        'noise_type': 'freq',
                                                                                                        'height': height, 
                                                                                                        'width': width,
                                                                                                        'dataset_size': dataset_size,
                                                                                                        'transform': transform, 
                                                                                                        'joint_transform': joint_transform, 
                                                                                                        'use_multi_crop': False, 
                                                                                                        'K_init': K_init if mode == 'train_init' else None,},

                                                                                    'randomUniformLowRes': {
                                                                                                        'num_classes': ds_num_classes,
                                                                                                        'noise_type': 'uniform',
                                                                                                        'height': height, 
                                                                                                        'width': width,
                                                                                                        'dataset_size': dataset_size,
                                                                                                        'transform': transform, 
                                                                                                        'joint_transform': joint_transform, 
                                                                                                        'use_multi_crop': False, 
                                                                                                        'K_init': K_init if mode == 'train_init' else None,},

                                                                                    'randomUniformGrayLowRes': {
                                                                                                        'num_classes': ds_num_classes,
                                                                                                        'noise_type': 'uniform',
                                                                                                        'height': height, 
                                                                                                        'width': width,
                                                                                                        'dataset_size': dataset_size,
                                                                                                        'transform': transform, 
                                                                                                        'joint_transform': joint_transform, 
                                                                                                        'use_multi_crop': False, 
                                                                                                        'K_init': K_init if mode == 'train_init' else None,},

                                                                                        'imagenet': {'root': imagenet_data_dir, 
                                                                                                        'train': (mode in ['train', 'train_init']), 
                                                                                                        'transform': transform, 
                                                                                                        'joint_transform': joint_transform, 
                                                                                                        'use_multi_crop': (mode=='train'), 
                                                                                                        'subset_txt': imagenet_subset_txt,
                                                                                                        'included_class_ratio': included_class_ratio,
                                                                                                        'K_init': K_init if mode == 'train_init' else None,},

                                                                                        'imagenetB': {'root': imagenet_data_dir, 
                                                                                                        'train': (mode in ['train', 'train_init']), 
                                                                                                        'transform': transform, 
                                                                                                        'joint_transform': joint_transform, 
                                                                                                        'use_multi_crop': (mode=='train'), 
                                                                                                        'subset_txt': imagenet_subset_txt,
                                                                                                        'included_class_ratio': included_class_ratio,
                                                                                                        'K_init': K_init if mode == 'train_init' else None,},

                                                                                                        
                                                                                        'imagenetGray': {'root': imagenet_data_dir, 
                                                                                                        'train': (mode in ['train', 'train_init']), 
                                                                                                        'transform': transform, 
                                                                                                        'joint_transform': joint_transform, 
                                                                                                        'use_multi_crop': (mode=='train'), 
                                                                                                        'subset_txt': imagenet_subset_txt,
                                                                                                        'included_class_ratio': included_class_ratio,
                                                                                                        'K_init': K_init if mode == 'train_init' else None,}
                                                                                                        }


    ds_name = dataset_name.rstrip('0123456789')

    ds_num_classes = dataset_name[len(ds_name):]
    if ds_num_classes == '':
        ds_num_classes = 0
    else:
        ds_num_classes = int(dataset_name[len(ds_name):])

    num_channels, height, width = DATASET_CONFIGS[ds_name]['input_dim']
    make_gray = DATASET_CONFIGS[ds_name]['make_gray']

    dataset_size = DATASET_CONFIGS[ds_name].get('dataset_size', None)

    transform = DATASET_CONFIGS[ds_name]['transform'][mode]

    dataset_cls = DATASET_CLASS[dataset_name]

    if backdoor_dict and backdoor_dict['backdoor'] and ds_num_classes > 0:

        bit_pixel_size = min(height, width) // backdoor_dict['pixels_per_bit']
        bit_pixel_size = max(bit_pixel_size, 1)  # Ensure at least 1 pixel size

        joint_transform = Backdoor(num_classes=ds_num_classes,
                                   height=height,
                                   width=width,
                                   bit_pixel_size=bit_pixel_size,
                                   marker_type=backdoor_dict['backdoor_type'],
                                   loc_type=backdoor_dict['backdoor_loc'],
                                   backdoor_noise_magnitude=backdoor_dict['backdoor_noise'],
                                   shuffle=backdoor_dict['shuffle'],
                                   backdoor_first=backdoor_dict['backdoor_first'],
                                   make_gray=make_gray,
                                   seed=seed)

        

    else:
        joint_transform = None
                            
   
    dataset_kwargs = DATASET_KWARGS(transform, joint_transform, height, width, dataset_size)[ds_name]

    dataset = dataset_cls(**dataset_kwargs)

    if sim_clr and mode == 'train':        
        ## modify transform before toTensor() for simclr objective


        def _get_color_distortion(p_distort=0.8): 
            # taken from https://github.com/dkobak/simclr-cifar10/

            color_distort = transforms.Compose([transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=p_distort), 
                                                transforms.RandomGrayscale(p=0.1)])

            return color_distort

        def _simclr_get_item(self, idx):
                img1, label = single_get_item(self, idx)
                img2, _ = single_get_item(self, idx)
                return [img1, img2], label


        totensor_idx = next((i for i, t in enumerate(dataset.transform.transforms) 
                             if isinstance(t, transforms.ToTensor)), None)


        ## replace transforms.RandomCrop(32, padding=4) w/ transforms.RandomResizedCrop(size=min(width, height), scale=(0.2, 1))
        random_crop_idx = next((i for i, t in enumerate(dataset.transform.transforms) 
                                if isinstance(t, transforms.RandomCrop)), None)
        
        if random_crop_idx is not None:
            dataset.transform.transforms[random_crop_idx] = transforms.RandomResizedCrop(size=min(width, height), 
                                                                                         scale=(0.2, 1.0))


        ## replace transforms.RandomResizedCrop(224) w/ transforms.RandomResizedCrop(size=min(width, height), scale=(0.2, 1))
        random_resized_crop_idx = next((i for i, t in enumerate(dataset.transform.transforms) 
                                        if isinstance(t, transforms.RandomResizedCrop)), None)
        
        if random_resized_crop_idx is not None:
            dataset.transform.transforms[random_resized_crop_idx] = transforms.RandomResizedCrop(size=min(width, height), 
                                                                                                 scale=(0.2, 1.0))

        dataset.transform.transforms.insert(totensor_idx, _get_color_distortion(p_distort=0.5),)


        ## overwrite __get_item__ to return two augmented views
        single_get_item = copy.deepcopy(dataset.__class__.__getitem__)

  
        
        dataset.__class__ = type(f"{type(dataset).__name__} {'Pair'}", (type(dataset),), {'__getitem__': _simclr_get_item})


    return dataset



def setup_data_loaders(args):

    # ==================================================
    # Data
    # ==================================================       
    train_dataset = get_dataset(args.dataset, mode='train', 
                                included_class_ratio=args.included_class_ratio,
                                backdoor_dict= {'backdoor': args.backdoor,
                                                'backdoor_type': args.backdoor_type,
                                                'backdoor_loc': args.backdoor_loc,
                                                'backdoor_noise': args.backdoor_noise,
                                                'pixels_per_bit': args.pixels_per_bit,
                                                'shuffle': False,
                                                'backdoor_first': args.backdoor_first,},

                                imagenet_data_dir=args.imagenet_data_dir,
                                ls_imagenet_data_dir=args.ls_imagenet_data_dir,
                                cifar_data_dir=args.cifar_data_dir,
                                imagenet_subset_txt=args.subset_txt,

                                mc_global_number=args.mc_global_number,
                                mc_global_scale=args.mc_global_scale,
                                mc_local_number=args.mc_local_number,
                                mc_local_scale=args.mc_local_scale,

                                seed=args.seed,
                                sim_clr=getattr(args, 'train_objective', None) == 'simclr',
                                )
    
    print('=> Training dataset:\n{}'.format(train_dataset))

    train_dataset_init = get_dataset(args.dataset, mode='train_init',  included_class_ratio=args.included_class_ratio,
                                     backdoor_dict= {'backdoor': args.backdoor, 
                                                     'backdoor_type': args.backdoor_type,
                                                     'backdoor_loc': args.backdoor_loc,
                                                     'backdoor_noise': args.backdoor_noise,
                                                     'pixels_per_bit': args.pixels_per_bit,
                                                     'shuffle': False,
                                                     'backdoor_first': args.backdoor_first,},
                                     imagenet_data_dir=args.imagenet_data_dir,
                                     ls_imagenet_data_dir=args.ls_imagenet_data_dir,
                                     cifar_data_dir=args.cifar_data_dir,
                                     imagenet_subset_txt=args.subset_txt,
                                     mc_global_number=args.mc_global_number,
                                     mc_global_scale=args.mc_global_scale,
                                     mc_local_number=args.mc_local_number,
                                     mc_local_scale=args.mc_local_scale,
                                     K_init=args.K_init,
                                     seed=args.seed,
                                     )
    
    print('=> Training dataset init:\n{}'.format(train_dataset_init))

    if "random" in args.dataset.lower():
        eval_dataset = args.dataset_ood
    else:   
        eval_dataset = args.dataset

    val_dataset = get_dataset(eval_dataset, mode='val',
                              backdoor_dict= {'backdoor': args.backdoor,
                                              'backdoor_type': args.backdoor_type,
                                              'backdoor_loc': args.backdoor_loc,
                                              'backdoor_noise': args.backdoor_noise,
                                              'pixels_per_bit': args.pixels_per_bit,
                                              'shuffle': False,
                                              'backdoor_first': args.backdoor_first,},
                              imagenet_data_dir=args.imagenet_data_dir,
                              ls_imagenet_data_dir=args.ls_imagenet_data_dir,
                              cifar_data_dir=args.cifar_data_dir,
                              imagenet_subset_txt=args.subset_txt,
                              mc_global_number=args.mc_global_number,
                              mc_global_scale=args.mc_global_scale,
                              mc_local_number=args.mc_local_number,
                              mc_local_scale=args.mc_local_scale,
                              seed=args.seed,
                              )
    
    print('=> Validation dataset:\n{}'.format(val_dataset))

    val_shuffle_dataset = get_dataset(eval_dataset, mode='val', 
                                backdoor_dict={'backdoor': args.backdoor,
                                               'backdoor_type': args.backdoor_type,
                                               'backdoor_loc': args.backdoor_loc,
                                               'backdoor_noise': args.backdoor_noise,
                                               'pixels_per_bit': args.pixels_per_bit,
                                               'shuffle': True,
                                               'backdoor_first': args.backdoor_first,},
                              imagenet_data_dir=args.imagenet_data_dir,
                              ls_imagenet_data_dir=args.ls_imagenet_data_dir,
                              cifar_data_dir=args.cifar_data_dir,
                              imagenet_subset_txt=args.subset_txt,
                              mc_global_number=args.mc_global_number,
                              mc_global_scale=args.mc_global_scale,
                              mc_local_number=args.mc_local_number,
                              mc_local_scale=args.mc_local_scale,
                              seed=args.seed,
                             )
    
    print('=> Validation shuffle dataset:\n{}'.format(val_shuffle_dataset))


    val_clean_dataset = get_dataset(eval_dataset, mode='val',  
                            backdoor_dict={'backdoor': False,},
                            imagenet_data_dir=args.imagenet_data_dir,
                            ls_imagenet_data_dir=args.ls_imagenet_data_dir,
                            cifar_data_dir=args.cifar_data_dir,
                            imagenet_subset_txt=args.subset_txt,
                            mc_global_number=args.mc_global_number,
                            mc_global_scale=args.mc_global_scale,
                            mc_local_number=args.mc_local_number,
                            mc_local_scale=args.mc_local_scale,
                            seed=args.seed)
    
    print('=> Validation clean dataset:\n{}'.format(val_clean_dataset))




    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               sampler=torch.utils.data.DistributedSampler(train_dataset, shuffle=True, seed=args.seed),
                                               batch_size=args.batch_size_per_gpu,
                                               num_workers=args.num_workers,
                                               pin_memory=True,
                                               drop_last=True,
                                               prefetch_factor=4,)
    

    train_init_loader = torch.utils.data.DataLoader(train_dataset_init,
                                                    sampler=torch.utils.data.DistributedSampler(train_dataset_init, shuffle=True, seed=args.seed),
                                                    batch_size=args.batch_size_per_gpu,
                                                    num_workers=args.num_workers,
                                                    pin_memory=True,
                                                    drop_last=False,
                                                    prefetch_factor=4,)
   
    
    val_clean_loader = torch.utils.data.DataLoader(val_clean_dataset,
                                                    batch_size=args.batch_size_per_gpu,
                                                    num_workers=args.num_workers,
                                                    shuffle=False, 
                                                    drop_last=False,
                                                    pin_memory=True,)
    
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                            batch_size=args.batch_size_per_gpu,
                                            num_workers=args.num_workers,
                                            shuffle=False, 
                                            drop_last=False,
                                            pin_memory=True,)


    if args.backdoor is not None:
        val_shuffle_loader = torch.utils.data.DataLoader(val_shuffle_dataset,
                                                        batch_size=args.batch_size_per_gpu,
                                                        num_workers=args.num_workers,
                                                        shuffle=False, 
                                                        drop_last=False,
                                                        pin_memory=True,)
    else:
        val_shuffle_loader = None


    if args.included_class_ratio is not None:
        val_fewer_classes_dataset = get_dataset(eval_dataset, mode='val', included_class_ratio=args.included_class_ratio,
                                            backdoor_dict={'backdoor': False,},
                                            imagenet_data_dir=args.imagenet_data_dir,
                                            ls_imagenet_data_dir=args.ls_imagenet_data_dir,
                                            cifar_data_dir=args.cifar_data_dir,
                                            imagenet_subset_txt=args.subset_txt,
                                            mc_global_number=args.mc_global_number,
                                            mc_global_scale=args.mc_global_scale,
                                            mc_local_number=args.mc_local_number,
                                            mc_local_scale=args.mc_local_scale,
                                            seed=args.seed)
        
        print('=> Validation fewer classes dataset:\n{}'.format(val_fewer_classes_dataset))

        val_fewer_classes_loader = torch.utils.data.DataLoader(val_fewer_classes_dataset,
                                                               batch_size=args.batch_size_per_gpu,
                                                               num_workers=args.num_workers,
                                                               shuffle=False, 
                                                               drop_last=False,
                                                               pin_memory=True,)
    else:
        val_fewer_classes_loader = None
    

    if args.dataset_ood  is not None:

        val_ood_dataset = get_dataset(args.dataset_ood, mode='val',
                                backdoor_dict={'backdoor': False,},
                                imagenet_data_dir=args.imagenet_data_dir,
                                ls_imagenet_data_dir=args.ls_imagenet_data_dir,
                                cifar_data_dir=args.cifar_data_dir,
                                imagenet_subset_txt=args.subset_txt,
                                mc_global_number=args.mc_global_number,
                                mc_global_scale=args.mc_global_scale,
                                mc_local_number=args.mc_local_number,
                                mc_local_scale=args.mc_local_scale,
                                seed=args.seed)
    
        print('=> Validation ood dataset:\n{}'.format(val_ood_dataset))

                            
        val_ood_loader = torch.utils.data.DataLoader(val_ood_dataset,
                                                    batch_size=args.batch_size_per_gpu,
                                                    num_workers=args.num_workers,
                                                    shuffle=False, 
                                                    drop_last=False,
                                                    pin_memory=True,)
    else:
        val_ood_loader = None

    val_iris_loader = None ## initialize iris val to None


    loader_dict = {'train_loader': train_loader,
                   'train_init_loader': train_init_loader,
                   'val_loader': val_loader,
                   'val_iris_loader': val_iris_loader,
                   'val_ood_loader': val_ood_loader,
                   'val_shuffle_loader': val_shuffle_loader,
                   'val_clean_loader': val_clean_loader,
                   'val_fewer_classes_loader': val_fewer_classes_loader}
    
    vars(args).update(loader_dict)

    return 


def setup_iris_loaders(args, model, iris_attack, return_iris_pair=False, train_iris=True):

    iris_dataset = get_dataset(args.dataset, mode='train_init' if train_iris else 'val',
                                    included_class_ratio=args.included_class_ratio,
                                    backdoor_dict= {'backdoor': args.backdoor,
                                                    'backdoor_type': args.backdoor_type,
                                                    'backdoor_loc': args.backdoor_loc,
                                                    'backdoor_noise': args.backdoor_noise,
                                                    'pixels_per_bit': args.pixels_per_bit,
                                                    'shuffle': False,
                                                    'backdoor_first': args.backdoor_first,},

                                    imagenet_data_dir=args.imagenet_data_dir,
                                    ls_imagenet_data_dir=args.ls_imagenet_data_dir,
                                    cifar_data_dir=args.cifar_data_dir,
                                    imagenet_subset_txt=args.subset_txt,

                                    mc_global_number=args.mc_global_number,
                                    mc_global_scale=args.mc_global_scale,
                                    mc_local_number=args.mc_local_number,
                                    mc_local_scale=args.mc_local_scale,
                                    K_init=args.K_iris,
                                    seed=args.seed,
                                    )
    


    iris_loader = torch.utils.data.DataLoader(iris_dataset,
                                              sampler=torch.utils.data.DistributedSampler(iris_dataset, shuffle= True if train_iris else False, seed=args.seed), 
                                              batch_size=args.batch_size_per_gpu,
                                              num_workers=args.num_workers,
                                              pin_memory=True,
                                              drop_last=False,
                                              prefetch_factor=4,)

    
    
    iris_dataset.setup_iris(model, iris_loader, iris_attack, save_dir=args.output_dir, 
                            return_iris_pair=return_iris_pair, precomp_iris_dir=args.iris_data_dir)
    
    if train_iris:

        ## overwrite the train_loader with iris data loader
        args.train_loader = torch.utils.data.DataLoader(iris_dataset,
                                                        sampler=torch.utils.data.DistributedSampler(iris_dataset, shuffle=True, seed=args.seed), 
                                                        batch_size=args.batch_size_per_gpu,
                                                        num_workers=args.num_workers,
                                                        pin_memory=True,
                                                        drop_last=True,
                                                        prefetch_factor=4,)
        print('=> Training dataset IRIs:\n{}'.format(args.train_loader.dataset))

    else:

        args.val_loader = torch.utils.data.DataLoader(iris_dataset,
                                                      sampler=torch.utils.data.DistributedSampler(iris_dataset, shuffle=False, seed=args.seed), 
                                                      batch_size=args.batch_size_per_gpu,
                                                      num_workers=args.num_workers,
                                                      pin_memory=True,
                                                      drop_last=False,
                                                      prefetch_factor=4,)

        
        print('=> Validation dataset IRIs:\n{}'.format(args.val_loader.dataset))

    return

def precompute_iris_data(args, model, iris_attack, return_iris_pair=False, train_iris=True):


    iris_dataset = get_dataset(args.dataset, mode='train_init' if train_iris else 'val',
                                    included_class_ratio=args.included_class_ratio,
                                    backdoor_dict= {'backdoor': args.backdoor,
                                                    'backdoor_type': args.backdoor_type,
                                                    'backdoor_loc': args.backdoor_loc,
                                                    'backdoor_noise': args.backdoor_noise,
                                                    'pixels_per_bit': args.pixels_per_bit,
                                                    'shuffle': False,
                                                    'backdoor_first': args.backdoor_first,},

                                    imagenet_data_dir=args.imagenet_data_dir,
                                    ls_imagenet_data_dir=args.ls_imagenet_data_dir,
                                    cifar_data_dir=args.cifar_data_dir,
                                    imagenet_subset_txt=args.subset_txt,

                                    mc_global_number=args.mc_global_number,
                                    mc_global_scale=args.mc_global_scale,
                                    mc_local_number=args.mc_local_number,
                                    mc_local_scale=args.mc_local_scale,
                                    K_init=args.K_iris,
                                    seed=args.seed,
                                    )
    

    iris_loader = torch.utils.data.DataLoader(iris_dataset,
                                              sampler=torch.utils.data.DistributedSampler(iris_dataset, shuffle=True, seed=args.seed), 
                                              batch_size=args.batch_size_per_gpu,
                                              num_workers=args.num_workers,
                                              pin_memory=True,
                                              drop_last=False,
                                              prefetch_factor=4,)


    if args.iris_data_dir is None:
        raise ValueError("Please provide a directory to save precomputed IRIs using --precomp_iris")

    iris_dataset.compute_iris(model, iris_loader, iris_attack, precomp_iris_dir=args.iris_data_dir,
                              return_iris_pair=return_iris_pair,)
    

    return

