import os
import random

from utils import fix_random_seeds

SEED = 0
PATH_TO_IMAGENET = '/path/to/imagenet/data/ImageNet_2012/train'
PATH_TO_IMAGENET1000 = './data/imagenet1000.txt'

fix_random_seeds(SEED)

imagenet_classes = os.listdir(PATH_TO_IMAGENET)


with open(PATH_TO_IMAGENET1000, 'w') as output_file:
    for class_id in imagenet_classes:
        output_file.write(class_id + '\n')