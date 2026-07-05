import os 
import sys 

sys.path.append(os.path.abspath("external"))
os.environ["WANDB__SERVICE_WAIT"] = "300" 

import argparse

import torch
import torch.nn as nn

from external.attacks import IrisPGD

from utils import get_args, init_distributed_mode, fix_random_seeds, print_program_info
from dataloaders import setup_data_loaders, precompute_iris_data
from models import make_frank_model, make_model, get_stitch_num_layers

 

def generate_iris(args):


    def _setup_output_directory(args):
        
        if "random" in args.dataset.lower():
            assert args.dataset_ood is not None , "For noise datasets, dataset_ood must be specified."
            assert args.included_class_ratio is None, "For noise datasets, there are not classes to include."

            dataset_eval = args.dataset_ood
            dataset_train = args.dataset
        else:
            dataset_eval = dataset_train = args.dataset

        _arch_front = args.arch_front + "_wo_res" if args.remove_residuals_front else args.arch_front
        _arch_end = args.arch_end + "_wo_res" if args.remove_residuals_end else args.arch_end


        args.front_id =  args.front_model_weights.split("/")[-4] + "_" + args.front_model_weights.split("/")[-2]
        args.end_id =  args.end_model_weights.split("/")[-4] + "_" + args.end_model_weights.split("/")[-2]


        
        base_path = [args.output_dir,
                     args.ablation_name,
                     args.experiment_mode,
                     dataset_eval,
                     dataset_train, 
                     f"{_arch_front}_{_arch_end}",
                     f"{args.front_id}_{args.end_id}",
                     f"{args.inverse_index}",
                     f"{args.init_transform}_K_init_{args.K_init}_DM_{args.direct_matching}_RS_{args.bn_running_stats}",
                     f"{args.seed}"]
       
        if args.iris:
            base_path[4] = base_path[4] + f"_iris_ratio_{args.iris_attack_ratio}_K_{args.K_iris}"

        if args.included_class_ratio is not None:
            base_path[4] = base_path[4] + f"_included_class_ratio_{args.included_class_ratio}"
               
        if args.robust is not None:
            base_path.insert(-2, f"atttack_{args.robust}_ratio_{args.attack_ratio}")

        if args.backdoor is not None:
            base_path.insert(-2, f"backdoor_{args.backdoor}_type_{args.backdoor_type}_loc_{args.backdoor_loc}_noise_{args.backdoor_noise}_first_{args.backdoor_first}")

        if not(args.direct_matching):
            objective_path = f"beta_cls_{args.beta_cls}_beta_soft_{args.beta_soft}_beta_functional_{args.beta_functional}_beta_hint_{args.beta_hint}"

            if args.beta_functional > 0:
                objective_path += f"_fs_{args.fula_soft}"
            
            if args.funct_cutoff is not None:
                objective_path += f"_funct_cutoff_{args.funct_cutoff}"
            base_path.insert(-1, objective_path)

        if args.iris_data_dir is not None:
            if args.backdoor is not None:
                _dataset = f'{dataset_eval}_backdoor_{args.backdoor}_type_{args.backdoor_type}_loc_{args.backdoor_loc}_noise_{args.backdoor_noise}_first_{args.backdoor_first}'
            else:
                _dataset = dataset_eval

            args.iris_data_dir = os.path.join(args.iris_data_dir, 'iris_data', _dataset, _arch_front, args.front_id, str(args.seed))


        args.output_dir = os.path.join(*base_path)
        os.makedirs(args.output_dir, exist_ok=True)

        print_program_info(args, os.path.join(args.output_dir, "program_info.txt"))

        return 

    _setup_output_directory(args)

    # load data
    setup_data_loaders(args)

    args.layers_to_stitch = get_stitch_num_layers(args.arch_end)


    args.front_layer_index = args.stitch_layer_index = args.inverse_index 
    generate_iris_layer(args)

    return 


def generate_iris_layer(args):

    print(f"Generating IRIs at layer {args.inverse_index}") 

    model = make_frank_model(arch_front=args.arch_front,
                             arch_end=args.arch_end,
                             
                             front_hook_post_relu=args.front_hook_post_relu,
                             end_hook_post_relu=args.end_hook_post_relu,
                             init_transform_mode=args.init_transform,
                             K_init=args.K_init,
                             bn_layers=args.bn_layers,
                             front_layer_index = args.front_layer_index,
                             stitch_layer_index=args.stitch_layer_index,
                             relative_front_index=args.relative_front_index,
                             weights_front=args.front_model_weights,
                             weights_end=args.end_model_weights,
                             multi_crop_wrapper=True,
                             data_loader=args.train_init_loader,
                             input_res=args.input_res,
                             remove_residuals_front=args.remove_residuals_front,
                             remove_residuals_end=args.remove_residuals_end,
                             num_classes_front=args.num_classes_front,
                             num_classes_end=args.num_classes_end,
                             )
    
   
    model = model.cuda()

    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], 
                                                find_unused_parameters=False)
    
          

    # ==================================================
    # Generating IRIS
    # ==================================================

    print("=> Constructing IRIs dataset ...")
    iris_attack = IrisPGD(eps=args.iris_epsilon,
                          random_start=True,
                          alpha=args.iris_step_size,
                          n_steps=args.iris_perturb_steps,
                          loss_fn=None,
                          min=args.iris_min,
                          max=args.iris_max,
                          attack_ratio=args.iris_attack_ratio,
                          stitch_layer_index=args.front_layer_index)

    precompute_iris_data(args, model, iris_attack, return_iris_pair=args.return_iris_pair)


    return 


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Training stitching models with YAML config")
    parser.add_argument(
        "--config_file",
        default="./configs/config_iris_generation.yaml",
        type=str,
        help="Path to the configuration YAML file."
    )
    parser.add_argument(
        "--local-rank",
        default=0,
        type=int,
        help="Please ignore this argument; No need to set it manually.",
    )

    cli_args = parser.parse_args()
    
    args = get_args(config_file=cli_args.config_file)

    init_distributed_mode(args)
    fix_random_seeds(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    # cudnn.benchmark = True

    if args.experiment_mode == "iris_generation":
        aggregate_metrics = generate_iris(args)
    else:
        raise ValueError(f"Experiment mode {args.experiment_mode} not supported.")

