import os
import random

from utils import fix_random_seeds

SEED = 0

PATH_TO_IMAGENET1000 = './data/imagenet1000.txt'
PATH_TO_IMAGENET100 = './data/imagenet100.txt'

PATH_TO_IMAGENETB1000 = './data/imagenetB100.txt'

fix_random_seeds(SEED)

with open(PATH_TO_IMAGENET1000, 'r') as file:
    class_ids_imagenet_1000 = file.read().splitlines()

with open(PATH_TO_IMAGENET100, 'r') as file:
    class_ids_imagenet_100 = file.read().splitlines()


class_ids = [cid for cid in class_ids_imagenet_1000 if cid not in class_ids_imagenet_100]


# Random subset of 100 classes
imagenetB100_classes = random.sample(class_ids, 100)


with open(PATH_TO_IMAGENETB1000, 'w') as output_file:
    for class_id in imagenetB100_classes:
        output_file.write(class_id + '\n')