import os
import yaml
import matplotlib.pyplot as plt
import numpy as np
import re
from matplotlib.lines import Line2D
import math
from PIL import Image
from pathlib import Path


def get_metric_from_folder(paths, ds, metric):

    is_cka = 'cka' in paths[0]
    if is_cka:
        metric = 'cka_score'


    def _column_averages_compact(lists):
        valid_lists = [lst for lst in lists if lst is not None]
        if not valid_lists:
            return []
        
        max_length = max(len(lst) for lst in valid_lists)
        
        return [
            sum(vals) / len(vals) if (vals := [
                row[i] for row in valid_lists 
                if i < len(row) and row[i] is not None
            ]) else None
            for i in range(max_length)
        ]

    yaml_paths = [os.path.join(file,'results.yaml') for file in paths]


    folder_metric = []

    for yaml_path in yaml_paths:


        with open(yaml_path, "r") as file:
            metrics = yaml.safe_load(file)


        num_layers = len(metrics)

        if is_cka:
            num_layers +=2

        current_metric = []
        for i in range(num_layers):
            try:
                if is_cka:
                    current_metric.append(round(float(metrics[i][metric]), 2)*100)
                else:
                    current_metric.append(metrics[i][ds][metric])
            except:
                current_metric.append(None) 

        folder_metric.append(current_metric)

    
    return _column_averages_compact(folder_metric), is_cka

def plot_results(result_dict, datasets_to_plot, metrics_to_plot, front_model_id=None,end_model_id=None, stitch_title=None, figure_name=None, plot_y=False):


    color_map = [ {'Real': '#1f77b4',
                   'IRIs': "#26601e",
                 }, 

                     {'Real': "#08a0f7",         # lighter blue
                      'IRIs': "#07fa3b",   # lighter green
                     }, 
                     
                     
                ]

    marker_map = {
                 'End/Front': 'X',
                  'TLM': '^',
                  'SLM': 'v', 
                  'FuLA': 'H', 
                  'Hint': '+',
                  }
    
    line_map =  { 
                  'End/Front':(0, (1, 5))  ,
                  'TLM': '-',
                  'SLM': '--', 
                  'FuLA': '-.', 
                  'Hint': ':',
                  }


    plt.figure(figsize=(11, 8))

    for metric_index, (data, metric) in enumerate(zip(datasets_to_plot, metrics_to_plot)):

        for mode_index, mode in enumerate(result_dict.keys()):

            for method, paths in result_dict[mode].items():
                metric_mean, is_cka = get_metric_from_folder(paths, data, metric)
                

                if is_cka:
                    ## CKA
                    pass
                else:
                    if metric_mean[-1] is not None:
                        plt.plot([len(metric_mean)-2, len(metric_mean)-1], [metric_mean[-2], metric_mean[-1]],  color='k',  linestyle=(0, (1, 5)))
                        plt.plot(len(metric_mean)-1, metric_mean[-1], marker='X', color='k',  markersize=14, lw=2.0)

                    plt.plot(0, metric_mean[0], marker='X', color='k', markersize=14, lw=2.0)
                    plt.plot([0, 1], [metric_mean[0], metric_mean[1]], color='k', linestyle=(0, (1, 5)))





                plt.plot(list(range(1, len(metric_mean)-1)), metric_mean[1:-1], marker=marker_map[method], linestyle=line_map[method], color=color_map[metric_index][mode],  markersize=14, lw=2.0)



            # Customize x-ticks with "Start" and "End"
            x_labels = [f'End \n {end_model_id}'] + list(range(1, len(metric_mean)-1)) + [f'Front \n {front_model_id}'] 
            plt.xticks(range(len(metric_mean)), x_labels, size=20)

            plt.xlabel('Stitching Layer', fontsize=24, labelpad=1) 

                



        mode_labels = list(color_map[metric_index].keys())
        mode_labels = [mode for mode in result_dict.keys()]

        mode_labels = [r'$\mathcal{D}^\text{IRIs}_\text{train}$' if label == 'IRIs' else label for label in mode_labels]
        mode_labels = [r'$\mathcal{D}_\text{train}$' if label == 'Real' else label for label in mode_labels]

        


        mode_handles = [Line2D([0], [0], color=color_map[metric_index][color_key], lw=7, markersize=16) for color_key in color_map[metric_index].keys() if color_key in result_dict.keys()]
        # Create custom legend for methods (markers and linestyles)
        method_handles = [Line2D([0], [0], color='black', marker=marker_map[m], linestyle=line_map[m], lw=4, markersize=16) for m in marker_map]
        method_labels = list(marker_map.keys())

        method_labels = ['DM'  if label == 'Hint' else label for label in method_labels]
        method_labels = ['HLM'  if label == 'TLM' else label for label in method_labels]



        # first_legend = plt.legend(mode_handles, mode_labels, loc='lower left', fontsize=16)
        # second_legend = plt.legend(method_handles, method_labels, loc='lower right', fontsize=16)

        first_legend_args = {'handles': mode_handles, 
                             'labels': mode_labels,}


        second_legend_args = {'handles': method_handles, 
                              'labels': method_labels,}



    y_label = 'Accuracy (%)' if metric == 'top1' else 'Agreement Probability (%)' if metric == 'agreement_prob' else None
    
    if plot_y:
        plt.ylabel(y_label, size=30)

    plt.yticks(range(0, 105, 20), fontsize=22)

    plt.ylim(0, 105)

    # plt.legend()
    if stitch_title is not None:
        plt.title(stitch_title, size=27)

    plt.savefig(os.path.join(paths[-1], figure_name),  bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close('all')

    list_of_paths.append(paths[-1])


    return first_legend_args, second_legend_args

if __name__ == '__main__':

    TLM_path = 'beta_cls_1.0_beta_soft_0.0_beta_functional_0.0_beta_hint_0.0'   
    SLM_path = 'beta_cls_0.0_beta_soft_1.0_beta_functional_0.0_beta_hint_0.0'
    FuLA_path = 'beta_cls_0.0_beta_soft_0.0_beta_functional_1.0_beta_hint_0.0_fs_False'
    Hint_path = 'beta_cls_0.0_beta_soft_0.0_beta_functional_0.0_beta_hint_1.0'
    CKA_path = 'cka'

    DATASET= 'cifar10'
    NUM_CLASSES = 10
    DIM = 'low'
    ARCH = 'resnet18_resnet18'
    # ARCH = 'vgg16_vgg16'
    # ARCH = 'vit_tiny_patch4_32_vit_tiny_patch4_32'

    # DIM = 'high'
    # DATASET='imagenet100'
    # NUM_CLASSES = 100
    # ARCH = 'resnet18_resnet18'
    # ARCH = 'vit_base_patch16_224_vit_base_patch16_224'


    DATASET_BACKDOOR = DATASET.replace(str(NUM_CLASSES), f'Pixel{NUM_CLASSES}')
    DATASET_SIMCLR = DATASET.replace(str(NUM_CLASSES), f'S.CLR{NUM_CLASSES}')

    if DIM == 'high':
       RESULT_PATH = f"PATH_TO_OUTPUT_DIR/stitching_models/corresponding_stitching/{DATASET}"
    else:
       RESULT_PATH = f"PATH_TO_OUTPUT_DIR/stitching_models/corresponding_stitching/{DATASET}"
    
    if ARCH == 'resnet18_resnet18':
        arch_title = 'ResNet-18'
    elif ARCH == 'vgg16_vgg16':
        arch_title = 'VGG-16'
    elif ARCH == 'vit_tiny_patch4_32_vit_tiny_patch4_32':
        arch_title = 'ViT-Tiny'
    elif ARCH == 'vit_base_patch16_224_vit_base_patch16_224':
        arch_title = 'ViT-Base'


    EXP_TO_RUN = {f'{DATASET}': [ 
        
                                 (f'{arch_title} \n Identical Stitching', f'{DATASET}_0', f'{DATASET}_0', 
                                 
                                                           [
                                                           (f"{ARCH}/{f'{DATASET}_0'}_{f'{DATASET}_0'}", 1),
                                                           (f"{ARCH}/{f'{DATASET}_0'}_{f'{DATASET}_0'}", 2),
                                                            (f"{ARCH}/{f'{DATASET}_0'}_{f'{DATASET}_0'}", 3)
                                                            ]),
                               
                               (f'{arch_title} \n Cross-seed Stitching', f'{DATASET}', f'{DATASET}_0',
                                                        [
                                                         (f"{ARCH}/{f'{DATASET}_1'}_{f'{DATASET}_0'}", 1),
                                                         (f"{ARCH}/{f'{DATASET}_2'}_{f'{DATASET}_0'}", 2),
                                                         (f"{ARCH}/{f'{DATASET}_3'}_{f'{DATASET}_0'}", 3)
                                                        ] ),
                               
  
                                (f'{arch_title} \n Cross-supervision Stitching', f'{DATASET_SIMCLR}', f'{DATASET}_0',
                                                        [
                                                        (f"{ARCH}/{f'{DATASET}_simclr_1'}_{f'{DATASET}_0'}", 1),
                                                        (f"{ARCH}/{f'{DATASET}_simclr_2'}_{f'{DATASET}_0'}", 2),
                                                        (f"{ARCH}/{f'{DATASET}_simclr_3'}_{f'{DATASET}_0'}", 3)
                                                            
                                                            ] ),

                                (f'{arch_title} \n Cross-bias Stitching', f'{DATASET_BACKDOOR}', f'{DATASET}_0',
                                                        [
                                                        (f"{ARCH}/{f'{DATASET}_backdoor_pixel_1'}_{f'{DATASET}_0'}", 1),
                                                        (f"{ARCH}/{f'{DATASET}_backdoor_pixel_2'}_{f'{DATASET}_0'}", 2),
                                                        (f"{ARCH}/{f'{DATASET}_backdoor_pixel_3'}_{f'{DATASET}_0'}", 3)
                                                         
                                                         ] ),


                                ],                                                                         
                }

    METRIC_TO_PLOT = ['agreement_prob']  # top1, agreement_prob
    
    DATA_MODES = ['Real', 'IRIs',]
    STITCHING_MODES = {'Real': ['TLM', 'SLM', 'FuLA', 'Hint',],
                       'IRIs': ['TLM', 'SLM', 'FuLA', 'Hint',],}
   
    

    for metric in METRIC_TO_PLOT:
        list_of_paths = []


        for experiment_id, (stitch_title, front_model_id, end_model_id, exp_path_list) in enumerate(EXP_TO_RUN[DATASET]):

            RESULT_DICT_OUTER = {}
            
            for exp_path, seed in exp_path_list:
                
                if 'backdoor' in exp_path:
                    EXP_DATA_PATHS = {'Real':  f"{RESULT_PATH}/{DATASET}/{exp_path}/backdoor_True_type_pixel_loc_rand_noise_0_first_False/pinv_K_init_500_DM_False_RS_True",
                                      'IRIs':  f"{RESULT_PATH}/{DATASET}_iris_ratio_1.0_K_None/{exp_path}/backdoor_True_type_pixel_loc_rand_noise_0_first_False/pinv_K_init_500_DM_False_RS_False",}                   
                else:

                    EXP_DATA_PATHS = {'Real': f"{RESULT_PATH}/{DATASET}/{exp_path}/pinv_K_init_500_DM_False_RS_True",
                                      'IRIs': f"{RESULT_PATH}/{DATASET}_iris_ratio_1.0_K_None/{exp_path}/pinv_K_init_500_DM_False_RS_False",}    

                for data_mode in DATA_MODES:

                    stitching_modes = STITCHING_MODES[data_mode]

                    for stitching_mode_ in stitching_modes:

                        if stitching_mode_ == 'TLM':
                            RESULT_DICT_OUTER.setdefault(data_mode, {}).setdefault('TLM', []).append(f'{EXP_DATA_PATHS[data_mode]}/{TLM_path}/{seed}') 
                        elif stitching_mode_ == 'SLM':
                            RESULT_DICT_OUTER.setdefault(data_mode, {}).setdefault('SLM', []).append(f'{EXP_DATA_PATHS[data_mode]}/{SLM_path}/{seed}')
                        elif stitching_mode_ == 'FuLA':
                            RESULT_DICT_OUTER.setdefault(data_mode, {}).setdefault('FuLA', []).append(f'{EXP_DATA_PATHS[data_mode]}/{FuLA_path}/{seed}')
                        elif stitching_mode_ == 'Hint':
                            RESULT_DICT_OUTER.setdefault(data_mode, {}).setdefault('Hint', []).append(f'{EXP_DATA_PATHS[data_mode]}/{Hint_path}/{seed}')
                
            first_legend_args, second_legend_args = plot_results(result_dict=RESULT_DICT_OUTER,
                                                        datasets_to_plot=['test_ind'],
                                                        metrics_to_plot=[metric],
                                                        front_model_id=front_model_id,end_model_id=end_model_id,
                                                        stitch_title=stitch_title,
                                                        figure_name=f'{metric}.png',
                                                        plot_y = (experiment_id == 0))
            

        # Collect all saved figure paths
        image_paths = [os.path.join(path, f'{metric}.png') for path in list_of_paths if os.path.exists(os.path.join(path, f'{metric}.png'))]

        if image_paths:
            images = [Image.open(img_path) for img_path in image_paths]
            n_images = len(images)
            n_rows = 1
            n_cols = math.ceil(n_images / n_rows)

            # Assume all images are the same size
            if len(images) > 1:
                img_width, img_height = images[1].size
            else:
                img_width, img_height = images[0].size
   
            img_width_first, img_height_first = images[0].size

            img_width_diff = img_width_first - img_width
            img_height_diff = img_height_first - img_height

            # Set spacing between images
            spacing_x = 40  # horizontal space in pixels
            spacing_y = 10  # vertical space in pixels

            # Add extra padding around the whole canvas
            padding_right = 55
            padding_bottom = 20
            padding_top = 20
            padding_left = 20

            # Create a blank canvas with extra space for spacing + padding
            combined_width = img_width_diff + (n_cols) * img_width + (n_cols - 1) * spacing_x + padding_left + padding_right
            combined_height = img_height_diff + (n_rows) * img_height + (n_rows - 1) * spacing_y + padding_top + padding_bottom
            combined_img = Image.new('RGB', (combined_width, combined_height), color=(255, 255, 255))

            for idx, img in enumerate(images):
                col = idx // n_rows
                row = idx % n_rows
                
                if col != 0:
                    x = padding_left + col * (img_width + spacing_x) + img_width_diff
                else:
                    x = padding_left + col * (img_width + spacing_x)
                if row != 1:
                    y = padding_top + row * (img_height + spacing_y) + img_height_diff
                else:
                    y = padding_top + row * (img_height + spacing_y)

                combined_img.paste(img, (x, y))



            # Create a matplotlib figure to add legends
            import matplotlib.pyplot as plt
            from matplotlib.patches import Rectangle
            
            fig, ax = plt.subplots(figsize=(combined_width/100, combined_height/100), dpi=100)
            ax.imshow(combined_img)
            ax.axis('off')
            
            
            # Estimate legend widths based on number of columns (rough approximation)
            legend1_width_estimate = len(first_legend_args['labels']) * 0.08  # ~0.08 per column
            legend2_width_estimate = len(second_legend_args['labels']) * 0.08

            # Total legend width including spacing between them
            legend_spacing = 0.01
            total_width = legend1_width_estimate + legend_spacing + legend2_width_estimate

            # Calculate the starting position (left edge of first legend)
            start_pos = (1.0 - total_width) / 2

            # Calculate center positions for each legend
            first_legend_x = start_pos + legend1_width_estimate / 2
            second_legend_x = start_pos + legend1_width_estimate + legend_spacing + legend2_width_estimate / 2

            # Fixed y-position in figure coordinates (e.g., 0.05 means 5% from bottom)
            legend_y = 0.05

            # Add the first legend (mode) using figure coordinates
            first_legend = ax.legend(**first_legend_args,
                                    loc='center',
                                    bbox_to_anchor=(first_legend_x, legend_y),
                                    bbox_transform=fig.transFigure,  # Use figure coordinates
                                    frameon=True,
                                    fancybox=False,
                                    shadow=False,
                                    fontsize=25,
                                    ncol=len(first_legend_args['labels']),
                                    markerscale=1.5,
                                    handlelength=2,
                                    handleheight=1.0,    
                                    labelspacing=0.5,    
                                    columnspacing=2.5,
                                    borderpad=0.5,      
                                    handletextpad=1.0)

            # Add the first legend to the figure so we can add a second one
            ax.add_artist(first_legend)

            # Add the second legend (method) using figure coordinates
            second_legend = ax.legend(**second_legend_args,
                                    loc='center',
                                    bbox_to_anchor=(second_legend_x, legend_y),
                                    bbox_transform=fig.transFigure,  # Use figure coordinates
                                    frameon=True,
                                    fancybox=False,
                                    shadow=False,
                                    fontsize=25,
                                    ncol=len(second_legend_args['labels']),
                                    markerscale=1.5,
                                    handlelength=4,
                                    handleheight=1.0,    
                                    labelspacing=0.5,    
                                    columnspacing=2.5,
                                    borderpad=0.5,      
                                    handletextpad=1.0)
                                
            # plt.tight_layout(pad=0)  


            os.makedirs(f'./construct_figures/figures/information_stitching/{ARCH}/{DATASET}', exist_ok=True)
        
            # Save with legends
            plt.savefig(f'./construct_figures/figures/information_stitching/{ARCH}/{DATASET}/{metric}.png', 
                        bbox_inches='tight', 
                        dpi=100,
                        facecolor='white')
