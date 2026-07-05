import sys

import math

import torch

from utils import *
from evaluation import accuracy
from training.losses import infoNCE_loss


def train_one_epoch(model, clf_loss, data_loader, optimizer, epoch, attack, args):
    metric_logger = MetricLogger(delimiter="  ")
    header = "Epoch: [{}/{}]".format(epoch, args.epochs)

    model.train()

    for it, (image, label) in enumerate(metric_logger.log_every(data_loader, 10, header)):

        it = len(data_loader) * epoch + it  # global training iteration

        # update weight decay and learning rate according to their schedule
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = args.lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = args.wd

        if isinstance(image, list):

            # move images to gpu
            image = [im.cuda(non_blocking=True) for im in image]

            # we have a list of images as a result of using multi-crop
            # so we repeat labels
            n_crops = args.mc_local_number + args.mc_global_number
            label = label.repeat(n_crops).cuda(non_blocking=True)
        else:

            # move images to gpu
            image = image.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)

        if attack is not None:
            ## robust training
            adv_image = attack.generate(model, image, label)          

            logit = model(adv_image)
        else:
            ## standard training
            logit = model(image)

        loss = clf_loss(logit, label)


        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)
       
        # parameter update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        # logging
        with torch.no_grad():
            acc1, acc5 = accuracy(logit.detach(), label, topk=(1, 5))
            metric_logger.update(top1=acc1.item())
            metric_logger.update(top5=acc5.item())
            metric_logger.update(loss=loss.item())
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
            metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    metric_dict = {f"{k}": meter.global_avg for k, meter in metric_logger.meters.items()}
    print("Averaged train stats:", metric_logger)

    return metric_dict


def train_one_epoch_simclr(model, projector, data_loader, optimizer, epoch, args):
    metric_logger = MetricLogger(delimiter="  ")
    header = "Epoch: [{}/{}]".format(epoch, args.ss_epochs)

    model.train()
    projector.train()

    for it, (image, label) in enumerate(metric_logger.log_every(data_loader, 10, header)):

        image1, image2 = image[0], image[1]

        it = len(data_loader) * epoch + it  # global training iteration

        # update weight decay and learning rate according to their schedule
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = args.simclr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = args.wd

        if isinstance(image1, list):

            # move images to gpu
            image1 = [im.cuda(non_blocking=True) for im in image1]
            image2 = [im.cuda(non_blocking=True) for im in image2]

            # we have a list of images as a result of using multi-crop
            # so we repeat labels
            n_crops = args.mc_local_number + args.mc_global_number
            label = label.repeat(n_crops).cuda(non_blocking=True)
        else:

            # move images to gpu
            image1 = image1.cuda(non_blocking=True)
            image2 = image2.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)


        _, feat = model(torch.cat((image1, image2), dim=0), return_feat='feat')

        rep = projector(feat)

        loss = infoNCE_loss(rep, args.simclr_temperature)

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)
       
        # parameter update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        # logging
        with torch.no_grad():
            metric_logger.update(loss=loss.item())
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
            metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    metric_dict = {f"{k}": meter.global_avg for k, meter in metric_logger.meters.items()}
    print("Averaged train stats:", metric_logger)


    return metric_dict