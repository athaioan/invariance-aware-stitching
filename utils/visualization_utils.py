import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.manifold import TSNE
from scipy.spatial.distance import pdist, squareform
from torch.nn import functional as F

import torch.distributed as dist

from utils import unwrap

from models import get_stitch_layer_name
# plt.rcParams['mathtext.fontset'] = 'cm' ## TODO: remove


def get_saliency_maps(args, model, dataloader, save_path=None):

    ## TODO: add support for different architectures and their combinations.

    def _get_misslabelled(model, image, label):
        
        logit = model(image)
    
        sample_loss = nn.CrossEntropyLoss(reduction='none')(logit, label)

        index = sample_loss.argmax().item()

        pred = logit.argmax(dim=1, keepdim=True)
        

        return index, label[index].item(), pred[index].item()

    def _plot_saliency(ax, image, attention_map):

        image = image.detach().cpu().numpy().transpose(1, 2, 0)
        attention_map = attention_map[0].detach().cpu().numpy()
        ax.imshow(image)
        ax.imshow(attention_map, cmap='jet', alpha=0.4)
        ax.axis('off')

        return 

    def _get_attention(model, image, index=None, power=2):

        if index is not None:
            image = image[index].unsqueeze(0)
        
        w, h = image.shape[-2], image.shape[-1]

        
        model.train()
        model.zero_grad()

        model(image)

        
        try:
            hooked_layers = model.hooked_activations
        except:
            hooked_layers = model.front_model.hooked_activations


        attention_maps = []

        for layer_index in range(len(hooked_layers)):

            attention_map = model.get_feature_map(layer_index)
            attention_map = (torch.abs(attention_map) ** power).sum(axis=1)  
            attention_map =  F.interpolate(attention_map.unsqueeze(1), size=(w, h), mode='bilinear', align_corners=False)

            attention_map_norm = torch.norm(attention_map.view(attention_map.size(0), -1), dim=1, keepdim=True)
            attention_map = attention_map / (attention_map_norm[:, None, None] + 1e-5)
            attention_map = attention_map.squeeze(1)

            attention_maps.append(attention_map)

        attention_maps = torch.stack(attention_maps, dim=0)

        return attention_maps

    def _get_grad_cam(model, image, index=None):


        if index is not None:
            image = image[index].unsqueeze(0)
        
        w, h = image.shape[-2], image.shape[-1]

        
        model.train()
        model.zero_grad()

        logit = model(image)

        pred = logit.argmax(dim=1, keepdim=True)
        logit_mean = (logit.gather(1, pred)).mean()

        logit_mean.backward()

        try:
            hooked_layers = model.hooked_activations
        except:
            hooked_layers = model.front_model.hooked_activations

        grad_cam_maps = []
        for layer_index in range(len(hooked_layers)):

            attention_map = model.get_feature_map(layer_index)
            gradient_map = attention_map.grad
            pooled_gradients = torch.mean(gradient_map, dim=[2, 3])

            activations = attention_map * pooled_gradients[:, :, None, None]

            heatmap = torch.mean(activations, dim=1)

            heatmap = torch.maximum(heatmap, torch.tensor(0))

            heatmap_max_norm = torch.max(heatmap.view(heatmap.size(0), -1), dim=1, keepdim=True)[0]

            heatmap = heatmap / heatmap_max_norm[None, :,:]

            heatmap =  F.interpolate(heatmap.unsqueeze(1), size=(w, h), mode='bilinear', align_corners=False)


            heatmap = heatmap.squeeze(1)

            grad_cam_maps.append(heatmap)

        grad_cam_maps = torch.stack(grad_cam_maps, dim=0)

        return grad_cam_maps
    
    # Create directory if it does not exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    model.train()

    iter_dataloader = iter(dataloader)
    image, label = next(iter_dataloader)

    if isinstance(image, list):
        image = [im.cuda(non_blocking=True) for im in image]
        n_crops = args.mc_local_number + args.mc_global_number
        label = label.repeat(n_crops).cuda(non_blocking=True)

    else:
        image = image.cuda(non_blocking=True)
        label = label.cuda(non_blocking=True)

    unwrap(model).reset_stitch_connection()
    index, gt, pred = _get_misslabelled(unwrap(model).end_model, image, label)


    layers_to_hook = range(0,len(unwrap(model).front_model.hooked_activations),2)


    fig = plt.figure(figsize=(25, 10))
    gs = gridspec.GridSpec(3, 2 * len(layers_to_hook), figure=fig, hspace=0.05, wspace=0.05) 
    axs = np.array([[fig.add_subplot(gs[i, j]) for j in range(2 * len(layers_to_hook))] for i in range(3)])

    attention_map_stitch = _get_attention(unwrap(model), image, index=index)
    grad_cam_map_stitch = _get_grad_cam(unwrap(model), image, index=index)

    attention_map_front = _get_attention(unwrap(model).front_model, image, index=index)
    grad_cam_map_front = _get_grad_cam(unwrap(model).front_model, image, index=index)

    unwrap(model).reset_stitch_connection()
    attention_map_end = _get_attention(unwrap(model).end_model, image, index=index)
    grad_cam_map_end = _get_grad_cam(unwrap(model).end_model, image, index=index)

    image[index] = image[index].flip(dims=[-1])

    for layer in layers_to_hook:
        

        axs[0, layer].set_title(f'Layer {layer} \n  Attention', fontsize=14)
        axs[0, layer+1].set_title(f'Layer {layer} \n  Grad-CAM', fontsize=14)
        _plot_saliency(axs[0, layer], image[index], attention_map_front[layer])
        _plot_saliency(axs[0, layer+1], image[index], grad_cam_map_front[layer])
        axs[0, 0].text(-0.5, 0.5, 'Front', rotation=90, va='center', ha='center', transform=axs[0, 0].transAxes, fontsize=14)

        _plot_saliency(axs[1, layer], image[index], attention_map_stitch[layer])
        _plot_saliency(axs[1, layer+1], image[index], grad_cam_map_stitch[layer])
        axs[1, 0].text(-0.5, 0.5, 'Stitch', rotation=90, va='center', ha='center', transform=axs[1, 0].transAxes, fontsize=14)

        _plot_saliency(axs[2, layer], image[index], attention_map_end[layer])
        _plot_saliency(axs[2, layer+1], image[index], grad_cam_map_end[layer])
        axs[2, 0].text(-0.5, 0.5, 'End', rotation=90, va='center', ha='center', transform=axs[2, 0].transAxes, fontsize=14)


    fig.suptitle(f'Saliency Maps \n GT: {gt} - Pred: {pred}', fontsize=24)
    plt.savefig(save_path)
    plt.close('all')  # Close figure to free memory


def plot_tsne(args, model, dataloader, num_tsne_samples=1000, num_plot_samples=100, labels_to_plot=None, save_path=None ):

    def _get_tsne_representations(activations, relative_space=False):

        activations = activations.reshape(activations.shape[0], -1)
        
        if relative_space:
            ## projection into relative space
            activations = pdist(activations)
            activations = squareform(activations)

        ## feature normalization
        activations = activations / (np.std(activations, axis=0, keepdims=True) + 1e-7)
        
        ## tsne
        tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, init='random', random_state=0,
                    n_jobs=-1)

        tsne_representation = tsne.fit_transform(activations)

        return tsne_representation

    front_activations = [[] for _ in range(len(get_stitch_layer_name(args.arch_front)))]
    end_activations = [[] for _ in range(len(get_stitch_layer_name(args.arch_end)))]
    stitch_activations = [[] for _ in range(len(get_stitch_layer_name(args.arch_end)))]
    image_labels = []
    
    # Create directory if it does not exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    iter_dataloader = iter(dataloader)
    image, label = next(iter_dataloader)

    count = 0

    for image, label in dataloader:

        if count > num_tsne_samples:
            break


        if isinstance(image, list):
            image = [im.cuda(non_blocking=True) for im in image]
            n_crops = args.mc_local_number + args.mc_global_number
            label = label.repeat(n_crops).cuda(non_blocking=True)

        else:
            image = image.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)

        if labels_to_plot is not None:
            indices = [i for i, lbl in enumerate(label) if lbl.item() in labels_to_plot]
            if not indices:
                continue
            image = image[indices]
            label = label[indices]

        else:
            pass

        count += len(label)

        ## labels
        image_labels.append(label.cpu().numpy())

        ## front model
        unwrap(model).reset_stitch_connection()
        unwrap(model).front_model(image)

        hooked_activations_front = {}
        for index, (key, value) in enumerate(unwrap(model).front_model.hooked_activations.items()):
            hooked_activations_front[key] = value.detach()
            front_activations[index].append(hooked_activations_front[key].cpu().numpy())

        ## end model
        unwrap(model).reset_stitch_connection()
        unwrap(model).end_model(image)

        hooked_activations_end = {}
        for index, (key, value) in enumerate(unwrap(model).end_model.hooked_activations.items()):
            hooked_activations_end[key] = value.detach()
            
            end_activations[index].append(hooked_activations_end[key].cpu().numpy())

        ## stitch model
        model(image)

        hooked_activations_stitched = {}

        for index, key in enumerate(unwrap(model).end_model.hooked_activations.keys()):

            if (index + 1) < unwrap(model).stitch_layer_index:
                hooked_activations_stitched[key] = unwrap(model).front_model.hooked_activations[key].detach()
            elif (index + 1) == unwrap(model).stitch_layer_index:
                hooked_activations_stitched[key] = unwrap(model).forced_output.detach()
            else:
                hooked_activations_stitched[key] = unwrap(model).end_model.hooked_activations[key].detach()

            stitch_activations[index].append(hooked_activations_stitched[key].cpu().numpy())

    
    layers_to_hook = [len(unwrap(model).front_model.hooked_activations)-3, 
                      len(unwrap(model).front_model.hooked_activations)-2,
                      len(unwrap(model).front_model.hooked_activations)-1,
                      ]
    
    fig = plt.figure(figsize=(25, 10))
    gs = gridspec.GridSpec(3, len(layers_to_hook), figure=fig, hspace=0.2, wspace=0.2)
    axs = np.array([[fig.add_subplot(gs[i, j]) for j in range(len(layers_to_hook))] for i in range(3)])


    image_labels = np.concatenate(image_labels, axis=0)[:num_tsne_samples]
   
    for index, layer_index in enumerate(layers_to_hook):

        front_activations[layer_index] = _get_tsne_representations(np.concatenate(front_activations[layer_index], axis=0)[:num_tsne_samples])
        end_activations[layer_index] = _get_tsne_representations(np.concatenate(end_activations[layer_index], axis=0)[:num_tsne_samples])
        stitch_activations[layer_index] = _get_tsne_representations(np.concatenate(stitch_activations[layer_index], axis=0)[:num_tsne_samples])

        # Determine the min and max limits for x and y based on all activations
        all_activations = np.concatenate([front_activations[layer_index][:num_plot_samples],
                                          stitch_activations[layer_index][:num_plot_samples],
                                          end_activations[layer_index][:num_plot_samples]
                                         ], axis=0)
        x_min, x_max = all_activations[:, 0].min(), all_activations[:, 0].max()
        y_min, y_max = all_activations[:, 1].min(), all_activations[:, 1].max()

        axs[0, index].set_title(f'Layer {layer_index+1}', fontsize=14)
        axs[0, index].scatter(front_activations[layer_index][:num_plot_samples, 0], front_activations[layer_index][:num_plot_samples, 1], marker='+', c=image_labels[:num_plot_samples], s=20)
        axs[0, index].set_xlim(x_min, x_max)
        axs[0, index].set_ylim(y_min, y_max)
        axs[0, index].set_aspect('equal')
        axs[0, index].axis('off')
        
        axs[1, index].scatter(stitch_activations[layer_index][:num_plot_samples, 0], stitch_activations[layer_index][:num_plot_samples, 1], marker='+', c=image_labels[:num_plot_samples], s=20)
        axs[1, index].set_xlim(x_min, x_max)
        axs[1, index].set_ylim(y_min, y_max)
        axs[1, index].set_aspect('equal')
        axs[1, index].axis('off')
        
        axs[2, index].scatter(end_activations[layer_index][:num_plot_samples, 0], end_activations[layer_index][:num_plot_samples, 1], marker='+', c=image_labels[:num_plot_samples], s=20)
        axs[2, index].set_xlim(x_min, x_max)
        axs[2, index].set_ylim(y_min, y_max)
        axs[2, index].set_aspect('equal')
        axs[2, index].axis('off')


        if index == 0:
            axs[0, index].text(-0.5, 0.5, 'Front', rotation=90, va='center', ha='center', transform=axs[0, 0].transAxes, fontsize=14)
            axs[1, index].text(-0.5, 0.5, 'Stitch', rotation=90, va='center', ha='center', transform=axs[1, 0].transAxes, fontsize=14)
            axs[2, index].text(-0.5, 0.5, 'End', rotation=90, va='center', ha='center', transform=axs[2, 0].transAxes, fontsize=14)
       
    fig.tight_layout()
    plt.savefig(save_path)
    plt.close('all') 

    return 


def plot_loss_landscape(args, model, loss_fn, dataloader, num_batches=15, 
                        num_points=20, alpha=1.0, save_path=None):
                        
    
    # Create directory if it does not exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Store original parameters
    original_params = [p.clone() for p in unwrap(model).transform.parameters()]
    
    ## filter normalization
    direction1, direction2 = [], []
    for p in unwrap(model).transform.parameters():
        
        direction1_ = torch.randn_like(p)
        direction2_ = torch.randn_like(p)

        if len(p.shape) > 1:
            direction1.append(direction1_ / torch.norm(direction1_, dim=0)[:, None] * torch.norm(p, dim=0)[:, None])
            direction2.append(direction2_ / torch.norm(direction2_, dim=0)[:, None] * torch.norm(p, dim=0)[:, None])
        else:
            direction1.append(direction1_ / torch.abs(direction1_) * torch.abs(p))
            direction2.append(direction2_ / torch.abs(direction2_) * torch.abs(p))

    
    # Create grid
    x = np.linspace(-alpha, alpha, num_points)
    y = np.linspace(-alpha, alpha, num_points)
    X, Y = np.meshgrid(x, y)
    
    # Calculate loss for each point
    Z = np.zeros_like(X)
    for i in range(num_points):
        for j in range(num_points):
            # Update model parameters
            for p, d1, d2 in zip(unwrap(model).transform.parameters(), direction1, direction2):
                p.data = p.data + X[i,j] * d1 + Y[i,j] * d2
            
            # Calculate loss
            total_loss = 0
            count_batches = 0
            for it, batch in enumerate(dataloader):
                
                if it == num_batches:
                    break
                
                image, label = batch

                

                if isinstance(image, list):

                    image = [im.cuda(non_blocking=True) for im in image]
                    n_crops = args.mc_local_number + args.mc_global_number
                    label = label.repeat(n_crops).cuda(non_blocking=True)

                else:

                    image = image.cuda(non_blocking=True)
                    label = label.cuda(non_blocking=True)
                    
                logit = model(image)


                loss = loss_fn(logit, label)
                total_loss += loss.item()
                count_batches += 1

            Z[i,j] = total_loss / num_batches
            
            # Reset model parameters
            for p, orig_p in zip(unwrap(model).transform.parameters(), original_params):
                p.data = orig_p.clone()
    
    if dist.get_rank() == 0:

        # Plot the loss landscape as a contour plot
        _, ax = plt.subplots(figsize=(10, 8))
        # CS = plt.contour(X, Y, Z, levels=20, cmap='summer')
        CS = plt.contour(X, Y, Z, levels=np.arange(0.1,10,0.5), cmap='summer')
        ax.set_xlabel('Direction 1')
        ax.set_ylabel('Direction 2')
        ax.set_title('Loss Landscape')
        ax.clabel(CS, CS.levels, inline=True, fontsize=10)

        plt.tight_layout()
        plt.savefig(save_path) 
        plt.close('all')
    
    return 

def plot_grad_histogram(grad_weight, grad_bias, num_bins=100, save_path=None):
    
    # Create directory if it does not exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if dist.get_rank() == 0:

        # Plot histograms of gradients
        _, axs = plt.subplots(1, 2, figsize=(12, 6))

        axs[0].hist(grad_weight, bins=num_bins, alpha=0.75, color='blue')
        axs[0].set_title('Weight Gradients Distribution')
        axs[0].set_xlabel('Gradient Value')
        axs[0].set_ylabel('Frequency')
        axs[0].grid(True)

        if grad_bias is not None:
            axs[1].hist(grad_bias, bins=num_bins, alpha=0.75, color='green')
            axs[1].set_title('Bias Gradients Distribution')
            axs[1].set_xlabel('Gradient Value')
            axs[1].set_ylabel('Frequency')
            axs[1].grid(True)


        plt.tight_layout()
        plt.savefig(save_path) 
        plt.close('all')

    return 

def plot_data_samples(args, num_samples=10, layer_front_index=None, layer_end_index=None, 
                                                                data_to_plot=['val', 'val_iris',
                                                                              'val_shuffle','val_ood', 'val_clean',
                                                                              'train', 'train_init'],
                                                                              reset=True):
    available_datasets = []
    all_data = []
    

    # reset DistributedSampler data_loaders
    if reset:
        for data_name in data_to_plot:
            data_loader = getattr(args, f"{data_name}_loader", None)
            if data_loader is not None and isinstance(data_loader.sampler, torch.utils.data.DistributedSampler):
                data_loader.sampler.set_epoch(0)          

    for data_name in data_to_plot:
        data_loader = getattr(args, f"{data_name}_loader", None)
        if data_loader is not None:
            available_datasets.append(data_name)
            data_loader = iter(data_loader)
            images, labels = next(data_loader)

            if len(images[0].shape) == 4:
                ## return_iris loader
                images = images[1]

           
            batch_size = len(images)
            samples_to_use = min(num_samples, batch_size)
            
            if isinstance(images, list):
                id_multi_crop = torch.randint(0, len(images), (1,)).item()
                images = images[id_multi_crop]
            
            all_data.append((data_name, images[:samples_to_use], labels[:samples_to_use]))
    
    if not available_datasets:
        print("No data loaders found!")
        return
    
    # Create subplot grid
    num_datasets = len(available_datasets)
    fig, axs = plt.subplots(num_datasets, num_samples, 
                           figsize=(num_samples * 2, num_datasets * 2))
    
    # Handle case where we only have one dataset or one sample
    if num_datasets == 1:
        axs = axs.reshape(1, -1)
    elif num_samples == 1:
        axs = axs.reshape(-1, 1)
    elif num_datasets == 1 and num_samples == 1:
        axs = axs.reshape(1, 1)
    
    for row_idx, (data_name, images, labels) in enumerate(all_data):
        actual_samples = len(images)
        
        for col_idx in range(num_samples):
            if num_datasets == 1 and num_samples == 1:
                ax = axs
            elif num_datasets == 1:
                ax = axs[0,col_idx]
            elif num_samples == 1:
                ax = axs[row_idx]
            else:
                ax = axs[row_idx, col_idx]
            
            if col_idx < actual_samples:
                # Display image
                img = images[col_idx].cpu().numpy().transpose(1, 2, 0)
                # Normalize image if needed (assuming it might be in [-1, 1] or [0, 1])
                if img.min() < 0:
                    print("Image pseudo-normalized for visualization")
                    img = (img + 1) / 2  # Convert from [-1, 1] to [0, 1]
                img = np.clip(img, 0, 1)
                
                ax.imshow(img)
                ax.set_title(f"Class: {labels[col_idx].item()}", fontsize=8) 
            else:
                # Empty subplot if we don't have enough samples
                ax.axis('off')
            
            ax.set_xticks([])
            ax.set_yticks([])
            
            ## Add row title (dataset name) only for the first column
            if col_idx == 0:
                ax.set_ylabel(data_name.replace('_', ' ').title(), 
                             rotation=90, labelpad=20, fontsize=10, fontweight='bold')

            # ## TODO: remove
            # ## $\mathcal{D}_\text{train}$'
            # if col_idx == 0:
            #     # ax.set_ylabel(r'$\mathcal{D}^\text{IRIs}_\text{train}$' + '\n' + r'$i = 9$', 
            #     ax.set_ylabel(r'$\mathcal{D}_\text{train}$', 
            #                  rotation=90, labelpad=20, fontsize=25, fontweight='bold')

    save_folder = os.path.join(args.output_dir, 'visualization')
    os.makedirs(save_folder, exist_ok=True)

    file_name = 'data.png' if args.experiment_mode == 'baseline' else \
        f'data_front_{layer_front_index}_end_{layer_end_index}.png' 

    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, file_name),  bbox_inches='tight', pad_inches=0.1, dpi=150)
    plt.close('all')

    return