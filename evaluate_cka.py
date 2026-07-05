import os 
import sys 

sys.path.append(os.path.abspath("external"))
os.environ["WANDB__SERVICE_WAIT"] = "300" 
 
import argparse


from external.cka_batch import CKA

import torch
import torch.nn as nn
import torch.distributed as dist

from external.attacks import PGD, IrisPGD

from utils import get_args, init_distributed_mode, fix_random_seeds, print_program_info, write_metrics_to_yaml, unwrap, plot_data_samples
from dataloaders import setup_data_loaders, setup_iris_loaders
from models import make_frank_model, make_model, get_stitch_num_layers
from models.model_utils import load_model


from tqdm import tqdm
import shutil


def run_corresponding_layer_cka(args):


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
                     args.metric_mode,
                     f"{args.seed}"]
       
        if args.iris:
            base_path[4] = base_path[4] + f"_iris_ratio_{args.iris_attack_ratio}_K_{args.K_iris}"

        if args.included_class_ratio is not None:
            base_path[4] = base_path[4] + f"_included_class_ratio_{args.included_class_ratio}"
               
        if args.robust is not None:
            base_path.insert(-2, f"atttack_{args.robust}_ratio_{args.attack_ratio}")

        if args.backdoor is not None:
            base_path.insert(-2, f"backdoor_{args.backdoor}_type_{args.backdoor_type}_loc_{args.backdoor_loc}_noise_{args.backdoor_noise}_first_{args.backdoor_first}")


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

    stitch_indices = [index for index in range(1, args.layers_to_stitch+1)]

    aggregate_metrics = {}
    for stitch_layer_index in stitch_indices: 

        args.front_layer_index = args.stitch_layer_index = stitch_layer_index
        aggregate_metrics[stitch_layer_index] = compute_metric(args)


    return aggregate_metrics


def compute_metric(args):

    print(f"Computing CKA between front layer {args.front_layer_index} to end layer {args.stitch_layer_index}") 


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
        if args.metric_mode == "cka":
            setup_iris_loaders(args, model, iris_attack, return_iris_pair=False, train_iris=False)
        else:
            setup_iris_loaders(args, model, iris_attack, return_iris_pair=True, train_iris=False)

    plot_data_samples(args, num_samples=6, layer_front_index=args.front_layer_index,
                      layer_end_index=args.stitch_layer_index)
    

    print("=> CKA computation starts ...")

    cka = CKA(unwrap(model), args.val_loader)

    if args.metric_mode == "cka":
        cka_score = cka.calculate_cka_matrix()[0]
    else:
        raise ValueError(f"Metric mode {args.metric_mode} not supported.")

    

    print(f"CKA score between front layer {args.front_layer_index} and end layer {args.stitch_layer_index}: {cka_score.item():.4f}")

    overall_stats = {"cka_score": float(cka_score.data.cpu().numpy()),}
                
    temp_data_dir = os.path.join(args.output_dir, 'temp_data')
    if os.path.isdir(temp_data_dir):
        shutil.rmtree(temp_data_dir)

    return overall_stats

    


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Training stitching models with YAML config")
    parser.add_argument(
        "--config_file",
        default="./configs/config_cka.yaml",
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
        aggregate_metrics = run_corresponding_layer_cka(args)
    else:
        raise ValueError(f"Experiment mode {args.experiment_mode} not supported.")


    write_metrics_to_yaml(aggregate_metrics, args.output_dir)
