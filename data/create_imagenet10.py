from utils import fix_random_seeds
import random

SEED = 0
PATH_TO_IMAGENET100 = './data/imagenet100.txt'
PATH_TO_IMAGENET10 = './data/imagenet10.txt'

fix_random_seeds(SEED)

with open(PATH_TO_IMAGENET100, 'r') as file:
    class_ids = file.read().splitlines()


# Random subset of 10 classes
selected_classes = random.sample(class_ids, 10)


with open(PATH_TO_IMAGENET10, 'w') as output_file:
    for class_id in selected_classes:
        output_file.write(class_id + '\n')