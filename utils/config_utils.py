
import os
import yaml
import argparse


def get_args(config_file):
    
    # Load YAML configurationg
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)

    config["subset_txt"] = os.path.join(config["subset_txt"], f'{config["dataset"].replace("Gray","")}.txt')

    # Convert dictionary to Namespace to mimic argparse
    args = argparse.Namespace(**config)

    return args