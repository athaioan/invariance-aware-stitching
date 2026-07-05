import sys

import numpy as np
import math

import torch

from utils import *
from evaluation import accuracy
from ..losses import compute_hint_loss, compute_functional_loss, compute_soft_target_loss


def train_one_epoch(model, clf_loss, data_loader, optimizer, epoch, attack, args):


    metric_logger = MetricLogger(delimiter="  ")
    header = "Epoch: [{}/{}]".format(epoch, args.epochs)


    if args.bn_running_stats:
        model.train()
    else:
        model.eval()


    for it, (image, label)  in enumerate(metric_logger.log_every(data_loader, 10, header)):
        
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


        if args.robust=="mandry":
            ## robust training
            adv_image = attack.generate(model, image, label)

            model_end_hooked_activations, end_logit = unwrap(model).get_end_hooked_activations(adv_image, train_mode=args.bn_running_stats)
            logit = model(adv_image)

        else:
            ## standard training
            model_end_hooked_activations, end_logit = unwrap(model).get_end_hooked_activations(image, train_mode=args.bn_running_stats)
            logit = model(image)    
        
                    
        classification_loss = clf_loss(logit, label)

        soft_targets_loss = compute_soft_target_loss(logit, end_logit)

        hint_loss = compute_hint_loss(unwrap(model).forced_output, unwrap(model).last_m2_out)
       
        functional_loss_stack, functional_loss = compute_functional_loss(unwrap(model).end_model.hooked_activations, 
                                                                         model_end_hooked_activations, 
                                                                         forced_output=unwrap(model).forced_output,
                                                                         last_m2_out=unwrap(model).last_m2_out,
                                                                         stitch_logit=logit,
                                                                         end_logit=end_logit,
                                                                         stitch_layer_index=unwrap(model).stitch_layer_index,
                                                                         funct_cutoff=args.funct_cutoff,
                                                                         fula_soft=args.fula_soft,)
        

        loss = args.beta_cls * classification_loss + args.beta_soft * soft_targets_loss + \
               args.beta_functional * functional_loss + args.beta_hint * hint_loss 


        if args.relative_betas: 
            loss = loss / (abs(args.beta_cls) + abs(args.beta_soft) + abs(args.beta_functional) + abs(args.beta_hint))
        
        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            print("Classification Loss: {}, Soft Targets Loss: {}, FuLA Loss: {}, Hint Loss: {}".format(
                  classification_loss.item(), soft_targets_loss.item(), functional_loss.item(), hint_loss.item()))

            sys.exit(1)

        if args.direct_matching:
            pass
        else:
            # parameter update
            optimizer.zero_grad()       
            loss.backward() 
            optimizer.step()

        # logging
        with torch.no_grad():

            ## accuracy
            acc1, acc5 = accuracy(logit.detach(), label, topk=(1, 5))

            metric_logger.update(loss=loss.item())
            metric_logger.update(top1=acc1.item())
            metric_logger.update(top5=acc5.item())

            metric_logger.update(classification_loss=classification_loss.item())
            metric_logger.update(soft_targets_loss=soft_targets_loss.item())
            metric_logger.update(functional_loss=functional_loss.item())
            metric_logger.update(hint_loss=hint_loss.item())

            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
            metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])



    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    metric_dict = {f"{k}": meter.global_avg for k, meter in metric_logger.meters.items() if 'grad' not in k}
    
    metric_dict["functional_loss_stack"] = functional_loss_stack.data.cpu().numpy().astype(np.float32).tolist() + [None]

    print("Averaged train stats:", metric_logger)

    return metric_dict

def train_one_epoch_sanity(model, clf_loss, data_loader, optimizer, epoch, attack, args):


    metric_logger = MetricLogger(delimiter="  ")
    header = "Epoch: [{}/{}]".format(epoch, args.epochs)


    if args.bn_running_stats:
        model.train()
    else:
        model.eval()


    for it, (images, label)  in enumerate(metric_logger.log_every(data_loader, 10, header)):
        
        image, image_iris = images[0], images[1]

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
            image_iris = image_iris.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)


        if args.robust=="mandry":
            ## robust training
            adv_image = attack.generate(model, image, label)

            model_end_hooked_activations, end_logit = unwrap(model).get_end_hooked_activations(adv_image, train_mode=args.bn_running_stats)
            logit = model(adv_image)

        else:

            if args.iris_end:
                ## standard training
                model_end_hooked_activations, end_logit = unwrap(model).get_end_hooked_activations(image_iris, train_mode=args.bn_running_stats)
                logit = model(image)    

            else:
                ## standard training
                model_end_hooked_activations, end_logit = unwrap(model).get_end_hooked_activations(image, train_mode=args.bn_running_stats)
                logit = model(image_iris)    
            
                    
        classification_loss = clf_loss(logit, label)

        soft_targets_loss = compute_soft_target_loss(logit, end_logit)

        hint_loss = compute_hint_loss(unwrap(model).forced_output, unwrap(model).last_m2_out)
       
        functional_loss_stack, functional_loss = compute_functional_loss(unwrap(model).end_model.hooked_activations, 
                                                                         model_end_hooked_activations, 
                                                                         forced_output=unwrap(model).forced_output,
                                                                         last_m2_out=unwrap(model).last_m2_out,
                                                                         stitch_logit=logit,
                                                                         end_logit=end_logit,
                                                                         stitch_layer_index=unwrap(model).stitch_layer_index,
                                                                         funct_cutoff=args.funct_cutoff,
                                                                         fula_soft=args.fula_soft,)
        

        loss = args.beta_cls * classification_loss + args.beta_soft * soft_targets_loss + \
               args.beta_functional * functional_loss + args.beta_hint * hint_loss 


        if args.relative_betas: 
            loss = loss / (abs(args.beta_cls) + abs(args.beta_soft) + abs(args.beta_functional) + abs(args.beta_hint))
        
        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        if args.direct_matching:
            pass
        else:
            # parameter update
            optimizer.zero_grad()       
            loss.backward() 
            optimizer.step()

        # logging
        with torch.no_grad():

            ## accuracy
            acc1, acc5 = accuracy(logit.detach(), label, topk=(1, 5))

            metric_logger.update(loss=loss.item())
            metric_logger.update(top1=acc1.item())
            metric_logger.update(top5=acc5.item())

            metric_logger.update(classification_loss=classification_loss.item())
            metric_logger.update(soft_targets_loss=soft_targets_loss.item())
            metric_logger.update(functional_loss=functional_loss.item())
            metric_logger.update(hint_loss=hint_loss.item())

            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
            metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])



    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    metric_dict = {f"{k}": meter.global_avg for k, meter in metric_logger.meters.items() if 'grad' not in k}
    
    metric_dict["functional_loss_stack"] = functional_loss_stack.data.cpu().numpy().astype(np.float32).tolist() + [None]

    print("Averaged train stats:", metric_logger)

    return metric_dict