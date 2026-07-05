import os
import copy

import numpy as np
from PIL import Image

from nltk.corpus import wordnet
from functools import partialmethod

import torch
from torchvision import transforms, datasets

from utils import unwrap
from tqdm import tqdm


def set_iris_mode(dataset_obj, return_iris_pair=False):

    
    def _getitem_iris(self, index, return_iris_pair=False):
       
        path, target = self.samples[index]
        path_iris, _ = self.samples_iris[index]

        img = self.loader(path)
        img_iris = self.loader(path_iris)

        if self.transform is not None:
            img, img_iris = self.transform(img), self.transform(img_iris)
    
        if return_iris_pair:
            return (img, img_iris), target
        else:
            return img_iris, target


    dataset_obj.joint_transform = None
    dataset_obj.target_transform = None
    dataset_obj.return_pair = return_iris_pair

    ## overwrite transform
    normalize_fn = None
    for t in dataset_obj.transform.transforms:
        if isinstance(t, transforms.Normalize):
            normalize_fn = t 

    if normalize_fn is not None:
        dataset_obj.transform = transforms.Compose([transforms.ToTensor(),
                                                    normalize_fn])
    else:
        dataset_obj.transform = transforms.Compose([transforms.ToTensor(),])

    iris_suffix = "(IRIs pairs)" if return_iris_pair else "(IRIs)"
    getitem_func = partialmethod(_getitem_iris, return_iris_pair=return_iris_pair)
    dataset_obj.__class__ = type(f"{type(dataset_obj).__name__} {iris_suffix}", (type(dataset_obj),), {'__getitem__': getitem_func})

    return



class ImageNet(datasets.ImageFolder):
    def __init__(self, root, train, transform=None, joint_transform=None, use_multi_crop=False, subset_txt=None, included_class_ratio=None, K_init=None, store_iris_path=None):
        
        self.root = os.path.join(root, 'train') if train else os.path.join(root, 'val')

        super(ImageNet, self).__init__(self.root, transform)

        self.joint_transform = joint_transform
        self.use_multi_crop = use_multi_crop

        if subset_txt is not None:
            with open(subset_txt, "r") as f:
                self.classes = f.read().splitlines()
                self.classes.sort()

            self.get_subset()

        self.get_idx_to_labels()


        if included_class_ratio is not None and included_class_ratio < 1.0:
            num_classes = len(self.classes)
            num_classes_to_include = int(num_classes * included_class_ratio)
            class_to_include = np.arange(num_classes)[:num_classes_to_include]
            idx_include = [i for i, target in enumerate(self.targets) if target in class_to_include]

            self.imgs = np.array(self.imgs)[idx_include].tolist()
            self.samples = list(zip(np.array(self.samples)[idx_include,0].tolist(),
                                    np.array(self.samples)[idx_include,1].astype(int).tolist()))
            self.targets = np.array(self.targets)[idx_include].tolist()

        if K_init is not None:

            idx_init = np.arange(len(self))
            np.random.shuffle(idx_init)
            idx_init = idx_init[:K_init]

            self.imgs = np.array(self.imgs)[idx_init].tolist()
            self.samples = list(zip(np.array(self.samples)[idx_init,0].tolist(),
                                    np.array(self.samples)[idx_init,1].astype(int).tolist()))
            self.targets = np.array(self.targets)[idx_init].tolist()


    def __getitem__(self, index):

        path, target = self.samples[index]
        img = self.loader(path)

        if self.joint_transform is not None and self.joint_transform.backdoor_first:
            img = self.joint_transform(img, target)

        if self.transform is not None:
            img = self.transform(img)
        
        if self.joint_transform is not None and not(self.joint_transform.backdoor_first):
            img = self.joint_transform(img, target)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
    



    def get_idx_to_labels(self):

        self.idx_to_labels = []
        for id in self.classes:
                        
            syns = wordnet.synset_from_pos_and_offset('n', int(id[1:]))
            lemmas = ", ".join(syns.lemma_names()).replace("_", " ")

            self.idx_to_labels.append(lemmas)

        return

    def get_subset(self):

        self.subset_class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}

        ## class to subset class mapping 
        set_to_subset = {}

        for class_label, idx in self.subset_class_to_idx.items():

            set_to_subset[self.class_to_idx[class_label]] = idx
                        
        self.samples_subset = []
        self.targets = []

        for index in range(len(self.samples)):
            path, class_index = self.samples[index]

            if class_index in set_to_subset.keys():
                self.samples_subset.append((path, set_to_subset[class_index]))
                self.targets.append(set_to_subset[class_index])


        ## updating subset variables
        self.class_to_idx = self.subset_class_to_idx

        self.samples = self.samples_subset
        self.imgs = self.samples_subset

        del self.subset_class_to_idx
        del self.samples_subset

        return
    

    def setup_iris(self, model, data_loader, iris_attack, save_dir=None, return_iris_pair=False, precomp_iris_dir=None):


        if precomp_iris_dir:
            self.root_before_iris = self.root
            self.root = os.path.join(precomp_iris_dir, str(iris_attack.stitch_layer_index))
        else:
            ## Generate iris
            self.compute_iris(model, data_loader, iris_attack, save_dir + f'/temp_data/iris_data', return_iris_pair=return_iris_pair)


        self.samples_iris = [(s[0].replace(self.root_before_iris, self.root), s[1]) for s in self.samples]

        if return_iris_pair:
            self.samples = [(s[0].replace('iris_data', 'regular_data'), s[1]) for s in self.samples_iris]


        self.targets = np.array(self.samples)[:,1].astype(int).tolist()
        self.imgs = [list(s) for s in self.samples]


        set_iris_mode(self, return_iris_pair=return_iris_pair)


        return 
    


    
    def compute_iris(self, model, data_loader, iris_attack, precomp_iris_dir, return_iris_pair=False):

        def _getitem_index(self, index):

            path, target = self.samples[index]
            img = self.loader(path)

            if self.joint_transform is not None and self.joint_transform.backdoor_first:
                img = self.joint_transform(img, target)

            if self.transform is not None:
                img = self.transform(img)
            
            if self.joint_transform is not None and not(self.joint_transform.backdoor_first):
                img = self.joint_transform(img, target)

            if self.target_transform is not None:
                target = self.target_transform(target)

            return img, target, index

        # overwrite __getitem__ function temporarily
        data_loader.dataset.__class__ = type(data_loader.dataset.__class__.__name__, 
                                            (data_loader.dataset.__class__,),
                                            {"__getitem__": _getitem_index})

        normalize_fn = None
        for t in data_loader.dataset.transform.transforms:
            if isinstance(t, transforms.Normalize):
                normalize_fn = t

        self.root_before_iris = self.root
        self.root = os.path.join(precomp_iris_dir, str(iris_attack.stitch_layer_index))


        for _, (image, _, indices) in enumerate(tqdm(data_loader, desc = "Generating IRIs")):

            iris_paths = [p.replace(self.root.replace(self.root, self.root_before_iris), self.root) for p in np.array(data_loader.dataset.samples)[indices,0]]

            image_paths = [p.replace('iris_data','regular_data') for p in iris_paths]

            ## Generate iris
            unwrap(model).reset_stitch_connection()
            iris_image = iris_attack.generate(unwrap(model), image, verbose=False)

            if normalize_fn is not None:
                
                image = image.to(iris_image.device)

                # Denormalize images
                normalize_mean = torch.tensor(normalize_fn.mean).view(1, 3, 1, 1).to(iris_image.device)
                normalize_std = torch.tensor(normalize_fn.std).view(1, 3, 1, 1).to(iris_image.device)

                iris_image = iris_image * normalize_std + normalize_mean
                iris_image = torch.clamp(iris_image, 0.0, 1.0) # clipping rounding errors 

                image = image * normalize_std + normalize_mean
                image = torch.clamp(image, 0.0, 1.0)  # clipping rounding errors              

            iris_image = (iris_image.cpu().numpy() * 255).astype(np.uint8).transpose(0, 2, 3, 1)
            iris_image = [Image.fromarray(img) for img in iris_image]

            for img, path in zip(iris_image, iris_paths):
                os.makedirs(os.path.dirname(path), exist_ok=True)
                img.save(path, format='PNG') 

            if return_iris_pair:

                image = (image.cpu().numpy() * 255).astype(np.uint8).transpose(0, 2, 3, 1)
                image = [Image.fromarray(img) for img in image]

                for img, path in zip(image, image_paths):
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    img.save(path, format='PNG') 

        return 