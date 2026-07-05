import os
import sys
import datetime

import torch
import torch.nn as nn
import torch.distributed as dist

def init_distributed_mode(args):
    # launched with torch.distributed.launch
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:

        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])

    # launched naively with "python main.py"
    # we manually add MASTER_ADDR to env variables
    elif torch.cuda.is_available():

        print("launched naively")

        print("==> Will run the code on one GPU.")
        args.rank, args.gpu, args.world_size = 0, 0, 1
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "12227"
    else:
        print("==> Does not support training without GPU.")
        sys.exit(1)

    dist.init_process_group(
        backend="nccl",
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank, 
        timeout=datetime.timedelta(days=1)
    )

    torch.cuda.set_device(args.gpu)
    print(
        "| distributed init (rank {}): {}".format(args.rank, args.dist_url), flush=True
    )
    dist.barrier()
    setup_for_distributed(args.rank == 0)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def unwrap(model):
    
    if isinstance(model, nn.parallel.DistributedDataParallel):

        return model.module.model
    
    else:
        
        return model.model