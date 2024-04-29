import torch
import os
import matplotlib.pyplot as plt

## for logger
import sys
from typing import Any, List, Tuple, Union



def _get_tensor_value(tensor):
  """
  Gets the value of a torch Tensor.
  it detaches the tensor from the graph, put it on cpu and convert it to numpy.
  """
  return tensor.cpu().detach().numpy()

def create_directories(parent_folder, training_params):
  
  """
  This functions gets a path to a parent folder and makes training-runs folder with following subfolders:
    - checkpoints
        -best_model
    - plots_imgs
  It also return path to these folders in the following order:
  Returns:
    current_run_folder, checkpoints_folder, best_folder, plots_imgs_folder
  """

  run_folders = os.path.join(parent_folder, 'training-runs')
  if not os.path.exists(run_folders):
     os.makedirs(run_folders)

  num_previous_runs = len(list(os.listdir(run_folders)))

  # run_number = 1 if num_previous_runs ==0 else num_previous_runs + 1
  # create a folder after the last run based on the run number not the number of folders in the folder
  last_runs_num = [int(i.split('_')[0]) for i in list(os.listdir(run_folders))]
  run_number = 1 if num_previous_runs ==0 else max(last_runs_num) +1

  num_zero = 3 - len(str(run_number)) # e.g., run_number  = 10 --> num_zero = 1
  pref_num = num_zero * str(0) + str(run_number) # e.g., 010

#   new_run_folder_name = f"{pref_num}_RES_{training_params['RES']}_ImgSize_{training_params['train_img_size']}_BatchSize_{training_params['batch_size']}_Lrg_{training_params['lrg']}_LrD_{training_params['lrd']}_"
  # If two different resolutions, then training_params['RES'] would be a list and causes problem in folder name:
  RES = training_params['RES'][0] if type(training_params['RES']) ==list else training_params['RES'] 
  new_run_folder_name = (
    f"{pref_num}_Loss_{training_params['loss']}_gamma{training_params['gamma']}_"
    f"NumDs{training_params['num_Ds']}_RES{RES}_"
    f"ImgSize_{training_params['train_img_size']}_"
    f"BatchSize{training_params['batch_size']}_"
    f"D_batch_size{training_params['D_batch_size']}_"
    f"Lrg{training_params['lrg']}_"
    f"LrD{training_params['lrd']}"
    # f"Lambda_{training_params['Lambda']}"
    )
  

  
  current_run_folder = os.path.join(run_folders, new_run_folder_name)
  if not os.path.exists(current_run_folder):
    os.makedirs(current_run_folder)

  checkpoints_folder = os.path.join(current_run_folder, 'checkpoints')
  if not os.path.exists(checkpoints_folder):
    os.makedirs(checkpoints_folder)

  best_folder = os.path.join(checkpoints_folder, 'best_model')
  if not os.path.exists(best_folder):
    os.makedirs(best_folder)

  plots_imgs_folder = os.path.join(current_run_folder, 'plots_imgs')
  if not os.path.exists(plots_imgs_folder):
    os.makedirs(plots_imgs_folder)

  return current_run_folder, checkpoints_folder, best_folder, plots_imgs_folder

def plot_image_grid(image_tensor, nrows=4, ncols = 4, figsize=(4, 4), title = 'x-direction', output_folder =None, file_name =None):
    '''
    Function for plotting a grid of images from a tensor.
    Assumes the input tensor has dimensions (num_images, height, width).
    '''
    fig, ax = plt.subplots(nrows=nrows, ncols=nrows, figsize=figsize)

    for i in range(nrows):
        for j in range(ncols):
            image = image_tensor[i * nrows + j]
            ax[i][j].imshow(image, cmap='gray')
            ax[i][j].axis('off')
    plt.suptitle(title)
    plt.tight_layout()
    if output_folder:
       plt.savefig(os.path.join(output_folder, f'{file_name}.png'), dpi =300)
    plt.close('all')
       

  


