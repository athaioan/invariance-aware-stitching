import os
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

target_class_file = './data/imagenet10.txt'

with open(target_class_file, 'r') as file:
    target_classes = file.read().splitlines()


imagenet_folder = '/path/to/imagenet/data/ImageNet_2012'
low_res_imagenet = '/path/to/imagenet/data/ImageNet_2012_low_res'

for mode in ['train', 'val']:
    split_folder_source = os.path.join(imagenet_folder, mode)

    image_labels = os.listdir(split_folder_source)

    for index_label, image_label in enumerate(image_labels):

        if image_label not in target_classes:
            continue

        image_label_folder = os.path.join(split_folder_source, image_label)

        image_files = os.listdir(image_label_folder)
        
        image_source_folder = os.path.join(low_res_imagenet, mode, image_label)
        os.makedirs(image_source_folder, exist_ok=True)

        for index, image_file in enumerate(image_files):


            print(f"Processing {index_label}/{len(image_labels)} {index}/{len(image_files)}")

            image_file_source_path = os.path.join(image_label_folder, image_file)
            image_file_target_path = os.path.join(low_res_imagenet, mode, image_label, image_file)

            resize_transform =  transforms.Compose([transforms.Resize(32),
                                                    transforms.CenterCrop(32),
                                                    ])
            
            image_orig = Image.open(image_file_source_path)
            image_resized = resize_transform(image_orig)

            if image_orig.mode == 'CMYK':
                image_resized = image_resized.convert('RGB')

            ## Saving as png to avoid jpeg lossy compression
            image_resized.save(image_file_target_path, format='PNG') 


