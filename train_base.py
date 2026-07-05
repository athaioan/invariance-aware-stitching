import os 
import argparse
import yaml
import shutil
import sys

sys.path.append(os.path.abspath("external"))
os.environ["WANDB__SERVICE_WAIT"] = "300"


import wandb
import sys

import torch
import torch.nn as nn
import torch.distributed as dist

from external.attacks import PGD

from utils import get_args, init_distributed_mode, fix_random_seeds, write_metrics_to_yaml, print_program_info, get_wandb_log, plot_data_samples, unwrap
from models import make_model, make_simclr_model

from training import step_scheduler, cosine_scheduler, get_params_groups
from training.base import train_one_epoch, train_one_epoch_simclr
from evaluation.base import eval, eval_robustness

from dataloaders import setup_data_loaders

def get_args(config_file="./configs/config.yaml"):
    # Load YAML configuration
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)


    config["subset_txt"] = os.path.join(config["subset_txt"], f'{config["dataset"]}.txt')
    dataset_arg = config["dataset"] + f"_backdoor_{config['backdoor_type']}" if config["backdoor"] else config["dataset"]
    dataset_arg = dataset_arg + "_adversarial" if config["robust"] else dataset_arg
    dataset_arg = dataset_arg + f"_{config['train_objective']}" if config["train_objective"] != "supervised" else dataset_arg

    config["output_dir"] = os.path.join(config["output_dir"], 
                                        config["ablation_name"],
                                        config["experiment_mode"],
                                        dataset_arg)
    
    # Convert dictionary to Namespace to mimic argparse
    args = argparse.Namespace(**config)

    return args

def train_supervised_baseline(args):

    def _setup_output_directory(args):

        _arch = args.arch + "_wo_res" if args.remove_residuals else args.arch

        args.output_dir = os.path.join(args.output_dir,
                                       _arch,
                                       f"{args.seed}")
        
        os.makedirs(args.output_dir, exist_ok=True)

        print_program_info(args, os.path.join(args.output_dir, "program_info.txt"))

        return


    _setup_output_directory(args)

    if dist.get_rank() == 0 and args.wandb:
        wandb_dir = os.path.join(args.output_dir, "wandb")
        os.makedirs(wandb_dir, exist_ok=True)
        wandb.init(project=args.wandb_project, reinit=True, dir=wandb_dir,
                   entity=args.wandb_entity, config=args,
                   settings=wandb.Settings(_disable_stats=True))
        

    model = make_model(args.arch, num_classes=args.num_classes, 
                       multi_crop_wrapper=True,
                       input_res=args.input_res,
                       remove_residuals=args.remove_residuals)
    
    model = model.cuda()
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], 
                                                find_unused_parameters=False)

    # ==================================================
    # Data
    # ==================================================
    setup_data_loaders(args)

    plot_data_samples(args, num_samples=6,  data_to_plot=['val', 'val_shuffle', 'train',])

    criterion = nn.CrossEntropyLoss().cuda()

    params_groups = get_params_groups(model, args.wd_regularize_all)


    if args.optimizer == "adam":

        optimizer = torch.optim.Adam(
            params_groups, lr=0,
        )  # we set lr and wd in train_one_epoch

    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            params_groups, lr=0, momentum=0.9, nesterov=args.nesterov
        )  # we set lr and wd in train_one_epoch

    else:

        raise ValueError("Optimizer not supported")


    if args.scheduler == "cosine":

        # cosine lr scheduler with linear warm-up
        args.lr_schedule = cosine_scheduler(
            args.lr
            * (args.batch_size_per_gpu * dist.get_world_size())
            / 256.0,  # linear scaling rule
            args.min_lr,
            args.epochs,
            len(args.train_loader),
            warmup_epochs=args.warmup_epochs,
        )
    
    elif args.scheduler == "step":
            
        args.lr_schedule = step_scheduler(args.lr,
                                               [int(1/3*args.epochs), int(2/3*args.epochs)],
                                               args.lr_decay,
                                               args.epochs,
                                               len(args.train_loader))
        
    else:
        
        raise ValueError("Scheduler not supported")
    


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
    print("=> Training starts ...")



    for epoch in range(0, args.epochs):

        args.train_loader.sampler.set_epoch(epoch)

        train_stats, test_stats = {}, {}


        # ============ training one epoch ... ============
        train_stats['ind'] = train_one_epoch(
            model, criterion, args.train_loader, optimizer, epoch, attack, args
        )

        # ============ evaluate model ... ============
        for key, loader, eval_func in [('ind', args.val_loader, eval),
                                       ('ind_shuffle', args.val_shuffle_loader, eval),
                                       ('robust', args.val_loader, eval_robustness)]:
            
            if loader and (key != 'robust' or (args.eval_robustness and (epoch % 50 == 0 or epoch == args.epochs - 1))):
                print(f"Evaluating {key} ...")
                test_stats[key] = eval_func(model, criterion, loader, epoch, args)


        overall_stats = {"train_ind": train_stats['ind'],
                         "test_ind": test_stats['ind'],
                         "test_ind_shuffle": test_stats.get('ind_shuffle', None),
                         "test_robust": test_stats.get('robust', None),}




        if dist.get_rank() == 0 and args.wandb:
            wandb_log = get_wandb_log(overall_stats)
            wandb.log(wandb_log, step=epoch)



        # ============ saving logs and model checkpoint ... ============
        save_dict = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1,
        }


        if dist.get_rank() == 0:
            torch.save(save_dict, os.path.join(args.output_dir, "checkpoint.pth"))
            if args.saveckpt_freq and epoch % args.saveckpt_freq == 0:
                shutil.copy(
                    os.path.join(args.output_dir, "checkpoint.pth"),
                    os.path.join(args.output_dir, f"checkpoint{epoch:04}.pth"),
                )

    return overall_stats

def train_simclr_baseline(args):

    def _setup_output_directory(args):

        _arch = args.arch + "_wo_res" if args.remove_residuals else args.arch

        args.output_dir = os.path.join(args.output_dir,
                                       _arch,
                                       f"{args.seed}")
        
        os.makedirs(args.output_dir, exist_ok=True)

        print_program_info(args, os.path.join(args.output_dir, "program_info.txt"))

        return


    _setup_output_directory(args)

    if dist.get_rank() == 0 and args.wandb:
        wandb_dir = os.path.join(args.output_dir, "wandb")
        os.makedirs(wandb_dir, exist_ok=True)
        wandb.init(project=args.wandb_project, reinit=True, dir=wandb_dir,
                   entity=args.wandb_entity, config=args,
                   settings=wandb.Settings(_disable_stats=True))
        

    model, projector = make_simclr_model(args.arch, num_classes=args.num_classes, 
                                         multi_crop_wrapper=True,
                                         input_res=args.input_res,
                                         remove_residuals=args.remove_residuals)
    
    model, projector = model.cuda(), projector.cuda()
    model, projector = nn.SyncBatchNorm.convert_sync_batchnorm(model), nn.SyncBatchNorm.convert_sync_batchnorm(projector)
    model, projector = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True), \
                       nn.parallel.DistributedDataParallel(projector, device_ids=[args.gpu], find_unused_parameters=False)

    # ==================================================
    # Data
    # ==================================================
    setup_data_loaders(args)

    plot_data_samples(args, num_samples=6,  data_to_plot=['val', 'val_shuffle', 'train',])

        
    cls_criterion = nn.CrossEntropyLoss().cuda()

    simclr_params_groups = get_params_groups(model, args.wd_regularize_all)
    projector_params_groups = get_params_groups(projector, args.wd_regularize_all)
    
    for group_index, pg in enumerate(projector_params_groups):
        simclr_params_groups[group_index]['params'].extend(pg['params'])

    cls_params_groups = get_params_groups(unwrap(model).get_classifier(), 
                                          args.wd_regularize_all)


    if args.optimizer == "adam":

        simclr_optimizer = torch.optim.Adam(
            simclr_params_groups, lr=0,
        )  # we set lr and wd in train_one_epoch

        cls_optimizer = torch.optim.Adam(
            cls_params_groups, lr=0,
        )  # we set lr and wd in train_one_epoch

    elif args.optimizer == "sgd":
        simclr_optimizer = torch.optim.SGD(
            simclr_params_groups, lr=0, momentum=0.9, nesterov=args.nesterov
        )  # we set lr and wd in train_one_epoch

        cls_optimizer = torch.optim.SGD(
            cls_params_groups, lr=0, momentum=0.9, nesterov=args.nesterov
        )  # we set lr and wd in train_one_epoch

    else:

        raise ValueError("Optimizer not supported")


    if args.scheduler == "cosine":

        # cosine lr scheduler with linear warm-up
        args.lr_schedule = cosine_scheduler(
            args.lr
            * (args.batch_size_per_gpu * dist.get_world_size())
            / 256.0,  # linear scaling rule
            args.min_lr,
            args.epochs,
            len(args.train_loader),
            warmup_epochs=args.warmup_epochs,
        )

        args.simclr_schedule = cosine_scheduler(
            args.lr
            * (args.batch_size_per_gpu * dist.get_world_size())
            / 256.0,  # linear scaling rule
            args.min_lr,
            args.ss_epochs,
            len(args.train_loader),
            warmup_epochs=args.warmup_epochs,
        )
    
    elif args.scheduler == "step":
            
        args.lr_schedule = step_scheduler(args.lr,
                                               [int(1/3*args.epochs), int(2/3*args.epochs)],
                                               args.lr_decay,
                                               args.epochs,
                                               
                                               len(args.train_loader))
        args.simclr_schedule = step_scheduler(args.lr,
                                               [int(1/3*args.ss_epochs), int(2/3*args.ss_epochs)],
                                               args.lr_decay,
                                               args.ss_epochs,
                                               len(args.train_loader))
        
    else:
        
        raise ValueError("Scheduler not supported")
    


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
    print("=> Training starts ...")



    for epoch in range(0, args.ss_epochs + args.epochs):

        if epoch == args.ss_epochs:
            # Revert pair mode
            args.train_objective = 'supervised'
            setup_data_loaders(args)

        args.train_loader.sampler.set_epoch(epoch)

        train_stats, test_stats = {}, {}

        if epoch < args.ss_epochs:
            # ============ training one ae epoch ... ============
            train_stats['ind'] = train_one_epoch_simclr(
                model, projector, args.train_loader, simclr_optimizer, epoch, args)
        else:
            # ============ training one cls epoch ... ============
            train_stats['ind'] = train_one_epoch(
                model, cls_criterion, args.train_loader, cls_optimizer, epoch - args.ss_epochs, attack, args)

            # ============ evaluate model ... ============
            for key, loader, eval_func in [('ind', args.val_loader, eval),
                                           ('ind_shuffle', args.val_shuffle_loader, eval),
                                           ('robust', args.val_loader, eval_robustness)]:
                
                if loader and (key != 'robust' or (args.eval_robustness and (epoch % 50 == 0 or epoch == args.epochs - 1))):
                    print(f"Evaluating {key} ...")
                    test_stats[key] = eval_func(model, cls_criterion, loader, epoch - args.ss_epochs, args)


        overall_stats = {"train_ind": train_stats['ind'],
                         "test_ind": test_stats.get('ind', None),
                         "test_ind_shuffle": test_stats.get('ind_shuffle', None),
                         "test_robust": test_stats.get('robust', None),}


        if dist.get_rank() == 0 and args.wandb:
            wandb_log = get_wandb_log(overall_stats)
            wandb.log(wandb_log, step=epoch)

        # ============ saving logs and model checkpoint ... ============
        save_dict = {
                     "model": model.state_dict(),
                     "optimizer": cls_optimizer.state_dict(),
                     "epoch": epoch + 1,}

        if dist.get_rank() == 0:
            torch.save(save_dict, os.path.join(args.output_dir, "checkpoint.pth"))
            if args.saveckpt_freq and epoch % args.saveckpt_freq == 0:
                shutil.copy(
                    os.path.join(args.output_dir, "checkpoint.pth"),
                    os.path.join(args.output_dir, f"checkpoint{epoch:04}.pth"),
                )

    return overall_stats

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Training base models with YAML config")
    parser.add_argument(
        "--config_file",
        default="./configs/config_baseline.yaml",
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

    if args.train_objective == 'supervised':
        overall_stats = train_supervised_baseline(args)
    elif args.train_objective == 'simclr':
        overall_stats = train_simclr_baseline(args)
    else:
        raise ValueError("Training objective not supported")


    write_metrics_to_yaml(overall_stats, args.output_dir)


