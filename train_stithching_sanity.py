import os 
import sys 

sys.path.append(os.path.abspath("external"))
os.environ["WANDB__SERVICE_WAIT"] = "300" 

import argparse

import wandb

import numpy as np

import torch
import torch.nn as nn
import torch.distributed as dist

from external.attacks import PGD, IrisPGD

from utils import (get_args, init_distributed_mode, fix_random_seeds, print_program_info, write_metrics_to_yaml, get_wandb_log, unwrap, freeze_bn_running_stats, unfreeze_bn_running_stats,
                  plot_data_samples)
from dataloaders import setup_data_loaders, setup_iris_loaders
from models import make_frank_model, make_model, get_stitch_num_layers
from models.model_utils import load_model

from training import step_scheduler, get_params_groups
from evaluation import compute_rank_affine, compute_rank_data
from training.stitching.trainer_pc import train_one_epoch_sanity
from evaluation.stitching import eval, eval_robustness 

import shutil


def run_corresponding_layer_stitching(args):


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

    stitch_indices = [index for index in range(args.layers_to_stitch+2)]

    aggregate_metrics = {}
    for stitch_layer_index in stitch_indices: 

        args.front_layer_index = args.stitch_layer_index = stitch_layer_index 

        args.skip_training = True if args.stitch_layer_index == stitch_indices[0] or args.stitch_layer_index == stitch_indices[-1] else False
        aggregate_metrics[stitch_layer_index] = train_stitching(args)

    return aggregate_metrics


def train_stitching(args):

    print(f"Stitching layer {args.front_layer_index} to layer {args.stitch_layer_index}") 


    if dist.get_rank() == 0 and args.wandb:
        wandb_dir = os.path.join(args.output_dir, "wandb")
        os.makedirs(wandb_dir, exist_ok=True)
        wandb.init(project=args.wandb_project, reinit=True, dir=wandb_dir,
                   entity=args.wandb_entity, config=args,
                   settings=wandb.Settings(_disable_stats=True))

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
    
    model_reference = make_model(args.arch_end, num_classes=args.num_classes_end, 
                                 weights=args.end_model_weights,
                                 multi_crop_wrapper=True,
                                 input_res=args.input_res,
                                 remove_residuals=args.remove_residuals_end)


    model_reference = model_reference.cuda()
    
    
    model = model.cuda()

    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], 
                                                find_unused_parameters=False)
    
    criterion = nn.CrossEntropyLoss().cuda()

    params_groups = get_params_groups(model, args.wd_regularize_all)

    if args.optimizer == "adam":
        
        optimizer = torch.optim.Adam(
            params_groups, lr=0,
        )  # we set lr and wd in train_one_epoch

    elif args.optimizer == "sgd":
        
        optimizer = torch.optim.SGD(
            params_groups, lr=0, momentum=0.9
        )  # we set lr and wd in train_one_epoch

    else:

        raise ValueError("Optimizer not supported")
    
    
    # step scheduler
    args.lr_schedule = step_scheduler(args.lr,
                                      [int(1/3*args.epochs), int(2/3*args.epochs)],
                                      args.lr_decay,
                                      args.epochs,
                                      len(args.train_loader))

    if args.robust == 'mandry':
        ## AT
        attack = PGD(eps=args.epsilon,
                     alpha=args.step_size,
                     n_steps=args.perturb_steps,
                     min=args.adv_min,
                     max=args.adv_max,
                     loss_fn=torch.nn.CrossEntropyLoss(),
                     attack_ratio=args.attack_ratio)
        
                  
    else: 
        ## Standard
        attack = None

       

    # ==================================================
    # Training
    # ==================================================

    if args.skip_training:
        
        print("=> Training skipped ...")

        # ============ evaluate model ... ============
        epoch = args.epochs - 1
        test_stats = {}

        for key, loader, eval_func in [('ind', args.val_loader, eval),                                       
                                       ('ind_shuffle', args.val_shuffle_loader, eval),
                                       ('ind_clean', args.val_clean_loader, eval),
                                       ('ind_fewer_classes', args.val_fewer_classes_loader, eval),
                                       ('ood', args.val_ood_loader, eval),
                                       ('robust', args.val_loader, eval_robustness)]:
            
            if loader and (key != 'robust' or args.eval_robustness):
                print(f"Evaluating {key} ...")
                test_stats[key] = eval_func(model, model_reference, criterion, loader, epoch, args)


        overall_stats = {"test_ind": test_stats.get('ind', None),
                         "test_ind_shuffle": test_stats.get('ind_shuffle', None),
                         "test_ind_clean": test_stats.get('ind_clean', None),
                         "test_ind_fewer_classes": test_stats.get('ind_fewer_classes', None),
                         "test_ood": test_stats.get('ood', None),
                         "test_robust": test_stats.get('robust', None),}
        

    else:


        if args.iris:
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
            
            setup_iris_loaders(args, model, iris_attack, return_iris_pair=True) 



        plot_data_samples(args, num_samples=6, layer_front_index=args.front_layer_index,
                          layer_end_index=args.stitch_layer_index,data_to_plot=['val', 'val_iris',
                                                                              'val_shuffle','val_ood', 'val_clean',
                                                                              'train', 'train_init'])
               
        print("=> Training starts ...")

        for epoch in range(0, args.epochs):

            args.train_loader.sampler.set_epoch(epoch)

            train_stats, test_stats = {}, {}


            if args.iris:
                # ============ training one epoch ... ============
                train_stats['ind'] = train_one_epoch_sanity(model, criterion, args.train_loader,
                                                            optimizer, epoch, attack, args)

            else:

                raise NotImplementedError("Iris training is required for this sanity check script.")


            # ============ evaluate model ... ============
            for key, loader, eval_func in [('ind', args.val_loader, eval),
                                           ('ind_iris', args.val_iris_loader, eval),
                                           ('ind_shuffle', args.val_shuffle_loader, eval),
                                           ('ind_clean', args.val_clean_loader, eval),
                                           ('ind_fewer_classes', args.val_fewer_classes_loader, eval),
                                           ('ood', args.val_ood_loader, eval),
                                           ('robust', args.val_loader, eval_robustness)]:
                
                if loader and (key != 'robust' or (args.eval_robustness and epoch == args.epochs - 1)):
                    print(f"Evaluating {key} ...")
                    test_stats[key] = eval_func(model, model_reference, criterion, loader, epoch, args)

            if dist.get_rank() == 0:

                ## get affine rank metrics
                for module in unwrap(model).transform.modules():
                    if type(module).__name__ == 'Conv2d':
                        affine_rank_stats = compute_rank_affine(
                                                module.weight[:, :, 0, 0],
                                                max_sigma_thold=1e-3)
                        
                    elif type(module).__name__ == 'Linear':
                        affine_rank_stats = compute_rank_affine(
                                                module.weight,
                                                max_sigma_thold=1e-3)
            

            overall_stats = {"train_ind": train_stats['ind'],
                             "test_ind": test_stats.get('ind', None),
                             "test_ind_iris": test_stats.get('ind_iris', None),
                             "test_ind_shuffle": test_stats.get('ind_shuffle', None),
                             "test_ind_clean": test_stats.get('ind_clean', None),
                             "test_ind_fewer_classes": test_stats.get('ind_fewer_classes', None),
                             "test_ood": test_stats.get('ood', None),
                             "test_robust": test_stats.get('robust', None),
                             "affine_rank_stats": affine_rank_stats,
                            }

            if dist.get_rank() == 0 and args.wandb:

                wandb_log = get_wandb_log(overall_stats)
                wandb.log(wandb_log, step=epoch)

                
        print("=> Training done.")
     

    temp_data_dir = os.path.join(args.output_dir, 'temp_data')
    if os.path.isdir(temp_data_dir):
        shutil.rmtree(temp_data_dir)
    

    return overall_stats


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Training stitching models with YAML config")
    parser.add_argument(
        "--config_file",
        default="./configs/config_sanity_stitch.yaml",
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

    if args.experiment_mode == "corresponding_stitching":
        aggregate_metrics = run_corresponding_layer_stitching(args)
    else:
        raise ValueError(f"Experiment mode {args.experiment_mode} not supported.")


    write_metrics_to_yaml(aggregate_metrics, args.output_dir)
