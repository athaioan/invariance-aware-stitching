import os
import copy

import numpy as np
from PIL import Image

from functools import partialmethod

import torch
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision import transforms


from utils import unwrap
from tqdm import tqdm

def set_iris_mode(dataset_obj, return_iris_pair=False):

    def _getitem_iris(self, index, return_iris_pair=False):

        img, img_iris, target = self.data[index], self.data_iris[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img, img_iris = Image.fromarray(img), Image.fromarray(img_iris)

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
    dataset_obj.transform = transforms.Compose([transforms.ToTensor(),])

    iris_suffix = "(IRIs pairs)" if return_iris_pair else "(IRIs)"
    getitem_func = partialmethod(_getitem_iris, return_iris_pair=return_iris_pair)
    dataset_obj.__class__ = type(f"{type(dataset_obj).__name__} {iris_suffix}", (type(dataset_obj),), {'__getitem__': getitem_func})


    return



class CIFAR10(CIFAR10):
    def __init__(self, root, train, download, transform, joint_transform, use_multi_crop=False, included_class_ratio=None, K_init=None):
        super(CIFAR10, self).__init__(root, train, transform, download=download) 

        self.joint_transform = joint_transform
        self.use_multi_crop = use_multi_crop

        if included_class_ratio is not None and included_class_ratio < 1.0:
            # Select a subset of classes
            num_classes = len(self.classes)
            num_classes_to_include = int(num_classes * included_class_ratio)
            class_to_include = np.arange(num_classes)[:num_classes_to_include]
            idx_include = [i for i, target in enumerate(self.targets) if target in class_to_include]
            self.data = self.data[idx_include]
            self.targets = np.array(self.targets)[idx_include].tolist()


        if K_init is not None:
            idx_init = np.arange(len(self))
            np.random.shuffle(idx_init)
            idx_init = idx_init[:K_init]

            self.data = self.data[idx_init]
            self.targets = np.array(self.targets)[idx_init].tolist()


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.joint_transform is not None and self.joint_transform.backdoor_first:
            img = self.joint_transform(img, target)

        if self.transform is not None:
            img = self.transform(img)

        if self.joint_transform is not None and not(self.joint_transform.backdoor_first):
            img = self.joint_transform(img, target)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


    def setup_iris(self, model, data_loader, iris_attack, save_dir=None, return_iris_pair=False, precomp_iris_dir=None, num_batches=5):

        def _load_batches(dataset_root):
                import pickle

                batch_files = sorted(os.listdir(dataset_root),  key=lambda x: int(x[-3:]))

                image_data = []
                labels = []
                indices = []

                for batch_file in batch_files:
                    with open(os.path.join(dataset_root, batch_file), 'rb') as f:
                        entry = pickle.load(f)

                        image_data = np.concatenate([image_data, entry['data']], axis=0) if len(image_data)>0 else entry['data']
                        labels.extend(entry['labels'])
                        indices.extend(entry['batch_indices'])

                image_data = image_data.reshape(-1, 3, 32, 32).transpose(0,2,3,1) # reshape and transpose to HWC

                sorted_indices = np.argsort(indices)
                image_data = image_data[sorted_indices]  
                labels = [labels[i] for i in sorted_indices]

                return image_data, labels


        if precomp_iris_dir is None:
            # temp directory to store iris data
            precomp_iris_dir = save_dir + f'/temp_data/iris_data'

            self.compute_iris(model, data_loader, iris_attack, precomp_iris_dir, 
                                return_iris_pair=return_iris_pair, num_batches=num_batches)
            

        iris_root = os.path.join(precomp_iris_dir, str(iris_attack.stitch_layer_index))
        regular_root = iris_root.replace('iris_data', 'regular_data')

        self.data_iris, self.targets  = _load_batches(iris_root)

        if return_iris_pair:
            self.data, _ = _load_batches(regular_root)
      
        set_iris_mode(self, return_iris_pair=return_iris_pair)

        return 


    def compute_iris(self, model, data_loader, iris_attack, precomp_iris_dir, return_iris_pair=False, num_batches=5):
            

            def _getitem_index(self, index):

                img, target = self.data[index], self.targets[index]

                if self.joint_transform is not None and self.joint_transform.backdoor_first:
                    img = self.joint_transform(img, target)

                if self.transform is not None:
                    img = self.transform(img)
                
                if self.joint_transform is not None and not(self.joint_transform.backdoor_first):
                    img = self.joint_transform(img, target)

                if self.target_transform is not None:
                    target = self.target_transform(target)

                return img, target, index

            def _store_batch(dataset_root, data_batch, labels_batch, indices_batch, count_batch):
                import pickle

                os.makedirs(dataset_root, exist_ok=True)
                ## store iris images
                batch_file = os.path.join(dataset_root, f'data_batch_{count_batch:03d}')

                with open(batch_file, 'wb') as f:
                    entry = {
                        'data': data_batch,
                        'labels': labels_batch,
                        'batch_label': f'batch {count_batch}',
                        'batch_indices': indices_batch,
                        'filenames': [],
                    }
                    pickle.dump(entry, f)

                return 
                       
            # overwrite __getitem__ function temporarily
            data_loader.dataset.__class__ = type(data_loader.dataset.__class__.__name__, 
                                                (data_loader.dataset.__class__,),
                                                {"__getitem__": _getitem_index})

            normalize_fn = None
            for t in data_loader.dataset.transform.transforms:
                if isinstance(t, transforms.Normalize):
                    normalize_fn = t

            iris_root = os.path.join(precomp_iris_dir, str(iris_attack.stitch_layer_index))
            regular_root = iris_root.replace('iris_data', 'regular_data')


            iris_batches, image_batches, label_batches, indices_batches, count_batch = [], [], [], [], 0

            num_store_batch = len(data_loader.dataset) // num_batches



            for it, (image, label, indices) in enumerate(tqdm(data_loader, desc = "Generating IRIs")):

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
                iris_image = iris_image.transpose(0,3,1,2).reshape(iris_image.shape[0], -1) # flatten

                image = (image.cpu().numpy() * 255).astype(np.uint8).transpose(0, 2, 3, 1)
                image = image.transpose(0,3,1,2).reshape(image.shape[0], -1) # flatten

                indices_batches.extend(indices.numpy().tolist())
                label_batches.extend(label.numpy().tolist())
                iris_batches =  np.concatenate([iris_batches, iris_image], axis=0) if len(iris_batches)>0 else iris_image
                image_batches =  np.concatenate([image_batches, image], axis=0) if len(image_batches)>0 else image


                if len(iris_batches) >= num_store_batch or (it+1)==len(data_loader):

                    count_batch += 1

                    num_to_store = len(iris_batches) if (it+1) == len(data_loader) else num_store_batch
                    
                    _store_batch(iris_root,  iris_batches[:num_to_store],  label_batches[:num_to_store], indices_batches[:num_to_store], count_batch)
                                       
                    if return_iris_pair:                      

                        _store_batch(regular_root,  image_batches[:num_to_store],  label_batches[:num_to_store], indices_batches[:num_to_store], count_batch)

                    iris_batches = iris_batches[num_to_store:]
                    image_batches = image_batches[num_to_store:]
                    label_batches = label_batches[num_to_store:]
                    indices_batches = indices_batches[num_to_store:]

            return

class CIFAR100(CIFAR100):
    def __init__(self, root, train, download, transform, joint_transform, use_multi_crop=False, included_class_ratio=None, K_init=None):
        super(CIFAR100, self).__init__(root, train, transform, download=download) 

        self.joint_transform = joint_transform
        self.use_multi_crop = use_multi_crop

        if included_class_ratio is not None and included_class_ratio < 1.0:
            # Select a subset of classes
            num_classes = len(self.classes)
            num_classes_to_include = int(num_classes * included_class_ratio)
            class_to_include = np.arange(num_classes)[:num_classes_to_include]
            idx_include = [i for i, target in enumerate(self.targets) if target in class_to_include]
            self.data = self.data[idx_include]
            self.targets = np.array(self.targets)[idx_include].tolist()


        if K_init is not None:
            idx_init = np.arange(len(self))
            np.random.shuffle(idx_init)
            idx_init = idx_init[:K_init]
            self.data = self.data[idx_init]
            self.targets = np.array(self.targets)[idx_init].tolist()


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.joint_transform is not None and self.joint_transform.backdoor_first:
            img = self.joint_transform(img, target)

        if self.transform is not None:
            img = self.transform(img)

        if self.joint_transform is not None and not(self.joint_transform.backdoor_first):
            img = self.joint_transform(img, target)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


    def setup_iris(self, model, data_loader, iris_attack, save_dir=None, return_iris_pair=False, precomp_iris_dir=None, num_batches=5):

            def _load_batches(dataset_root):
                    import pickle

                    batch_files = sorted(os.listdir(dataset_root),  key=lambda x: int(x[-3:]))

                    image_data = []
                    labels = []
                    indices = []

                    for batch_file in batch_files:
                        with open(os.path.join(dataset_root, batch_file), 'rb') as f:
                            entry = pickle.load(f)

                            image_data = np.concatenate([image_data, entry['data']], axis=0) if len(image_data)>0 else entry['data']
                            labels.extend(entry['labels'])
                            indices.extend(entry['batch_indices'])

                    image_data = image_data.reshape(-1, 3, 32, 32).transpose(0,2,3,1) # reshape and transpose to HWC

                    sorted_indices = np.argsort(indices)
                    image_data = image_data[sorted_indices]  
                    labels = [labels[i] for i in sorted_indices]

                    return image_data, labels


            if precomp_iris_dir is None:
                # temp directory to store iris data
                precomp_iris_dir = save_dir + f'/temp_data/iris_data'

                self.compute_iris(model, data_loader, iris_attack, precomp_iris_dir, 
                                    return_iris_pair=return_iris_pair, num_batches=num_batches)
                

            iris_root = os.path.join(precomp_iris_dir, str(iris_attack.stitch_layer_index))
            regular_root = iris_root.replace('iris_data', 'regular_data')

            self.data_iris, self.targets  = _load_batches(iris_root)

            if return_iris_pair:
                self.data, _ = _load_batches(regular_root)
        
            set_iris_mode(self, return_iris_pair=return_iris_pair)

            return 


    def compute_iris(self, model, data_loader, iris_attack, precomp_iris_dir, return_iris_pair=False, num_batches=5):
            

            def _getitem_index(self, index):

                img, target = self.data[index], self.targets[index]

                if self.joint_transform is not None and self.joint_transform.backdoor_first:
                    img = self.joint_transform(img, target)

                if self.transform is not None:
                    img = self.transform(img)
                
                if self.joint_transform is not None and not(self.joint_transform.backdoor_first):
                    img = self.joint_transform(img, target)

                if self.target_transform is not None:
                    target = self.target_transform(target)

                return img, target, index

            def _store_batch(dataset_root, data_batch, labels_batch, indices_batch, count_batch):
                import pickle

                os.makedirs(dataset_root, exist_ok=True)
                ## store iris images
                batch_file = os.path.join(dataset_root, f'data_batch_{count_batch:03d}')

                with open(batch_file, 'wb') as f:
                    entry = {
                        'data': data_batch,
                        'labels': labels_batch,
                        'batch_label': f'batch {count_batch}',
                        'batch_indices': indices_batch,
                        'filenames': [],
                    }
                    pickle.dump(entry, f)

                return 
                       
            # overwrite __getitem__ function temporarily
            data_loader.dataset.__class__ = type(data_loader.dataset.__class__.__name__, 
                                                (data_loader.dataset.__class__,),
                                                {"__getitem__": _getitem_index})

            normalize_fn = None
            for t in data_loader.dataset.transform.transforms:
                if isinstance(t, transforms.Normalize):
                    normalize_fn = t

            iris_root = os.path.join(precomp_iris_dir, str(iris_attack.stitch_layer_index))
            regular_root = iris_root.replace('iris_data', 'regular_data')


            iris_batches, image_batches, label_batches, indices_batches, count_batch = [], [], [], [], 0

            num_store_batch = len(data_loader.dataset) // num_batches



            for it, (image, label, indices) in enumerate(tqdm(data_loader, desc = "Generating IRIs")):

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
                iris_image = iris_image.transpose(0,3,1,2).reshape(iris_image.shape[0], -1) # flatten

                image = (image.cpu().numpy() * 255).astype(np.uint8).transpose(0, 2, 3, 1)
                image = image.transpose(0,3,1,2).reshape(image.shape[0], -1) # flatten

                indices_batches.extend(indices.numpy().tolist())
                label_batches.extend(label.numpy().tolist())
                iris_batches =  np.concatenate([iris_batches, iris_image], axis=0) if len(iris_batches)>0 else iris_image
                image_batches =  np.concatenate([image_batches, image], axis=0) if len(image_batches)>0 else image


                if len(iris_batches) >= num_store_batch or (it+1)==len(data_loader):

                    count_batch += 1

                    num_to_store = len(iris_batches) if (it+1) == len(data_loader) else num_store_batch
                    
                    _store_batch(iris_root,  iris_batches[:num_to_store],  label_batches[:num_to_store], indices_batches[:num_to_store], count_batch)
                                       
                    if return_iris_pair:                      

                        _store_batch(regular_root,  image_batches[:num_to_store],  label_batches[:num_to_store], indices_batches[:num_to_store], count_batch)

                    iris_batches = iris_batches[num_to_store:]
                    image_batches = image_batches[num_to_store:]
                    label_batches = label_batches[num_to_store:]
                    indices_batches = indices_batches[num_to_store:]

            return
    