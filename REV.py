''''
This script runs representative elementary volume (REV) analysis on a XCT volume.

it takes path to the XCT tif file and a list of image sizes, and number of samples as inputs  to caluclate:
two-point correlation function (S2) and its scaled version  (F2) in 3D.
These values can then be analyzed to REV for that image. 

Note: in windows when you pass the arguments in terminal, there is no need to have the path inside ''.
at least in my system it gives an error

'''
import os
import argparse
import numpy as np
import pandas as pd
import tifffile
from numba import jit
from tqdm import tqdm 
from typing import List, Dict
import joblib


from src.SMD_cal import calculate_two_point_3D, cal_fn

def REV(image: np.ndarray,
            img_size_list: List[int],
            n_rand_samples: int)-> Dict[str, np.ndarray]:
    
    """
    This function receives a 3D (XCT) image and calculates average S2 and F2 for a number of random subvolumes.
    These average correlation functions can then be analysed to determine the REV for the image.
    Parameters
    ----------
    image: np.ndarray
    This is the 3D image read as numpy array to do REV analysis on.

    img_size_list: List
    list of image sizes to calculate correlation functions. These sizes should be smaller than the whole image.

    n_rand_samples: int
    number of random images used for calculating REV. use 30 or more.

    Returns
    --------
    It returns two dictionary: one for s2 (s2_3d_dict) and one for f2 (f2_3d_dict)
"""
    
    x_max, y_max, z_max = image.shape[:]
    
    s2_3d_dict = {}
    f2_3d_dict = {}
    for image_size in tqdm(img_size_list):

        all_crops = np.zeros((n_rand_samples, image_size, image_size, image_size), dtype = np.uint8)
        for i in range(n_rand_samples):

            x = np.random.randint (0, x_max - image_size)
            y = np.random.randint (0, y_max - image_size)
            z = np.random.randint (0, z_max - image_size)

            crop_image = image[x:x + image_size, y:y + image_size, z:z + image_size]
            all_crops[i] = crop_image
            
        df_s2, df_f2 = calculate_two_point_3D(all_crops)
        s2_3d_dict[f'sub_{image_size}'] = df_s2
        f2_3d_dict[f'sub_{image_size}'] = df_f2

        print(f'{image_size} done !')
        
    return s2_3d_dict, f2_3d_dict

def parse_args():
  """Parses arguments."""
  parser = argparse.ArgumentParser()
  parser.add_argument('--image_dir', required= True, type=str,
                       help='Full path to the large image file. it should be .tif file')
  parser.add_argument('--image_sizes', required =True, type=int, nargs='+',
                      help='List of the image sizes you want to do REV analysis with')
  parser.add_argument('--n_rnd_samples', type=int, default=50,
                      help='Number of random subvolume of each size to calculate average S2, F2')
  parser.add_argument('--seed', type= int, default= 33)
  parser.add_argument('--output_dir', type=str,
                      help='Directory to save the results')
  return parser.parse_args()


def run_REV():
    args = parse_args()

    # print(args.image_dir)
    # print(args.img_size_list)
    # print(args.output_dir)
    np.random.seed(args.seed)

    # image_path = os.path.abspath(args.image_dir)
    # image_path = os.path.join(args.image_dir, 'Berea_11.4um_real_binary_512.tif')
    original_img = tifffile.imread(args.image_dir).astype(np.uint8)
    # check if it is binary: 0:solid, 1: pore
    if original_img.max() > 1:
        original_img = np.where(original_img > 1,1,0)

    # print(original_img.shape)
    # first calculate s2 and f2 for the whole image


    ## run REV function on the list of image

    s2_3d_dict, f2_3d_dict = REV(original_img, img_size_list = args.image_sizes,
                                    n_rand_samples=args.n_rnd_samples)
    


    _, _, _, s2_avg_original = calculate_two_point_3D(original_img)
    f2_avg_original = cal_fn(s2_avg_original, n = 2)

    s2_3d_dict['original'] = s2_avg_original
    f2_3d_dict['original'] = f2_avg_original

    joblib.dump(s2_3d_dict, os.path.join(args.output_dir, 's2_3d_dict.pkl'))
    joblib.dump(f2_3d_dict, os.path.join(args.output_dir, 'f2_3d_dict.pkl'))
    


if __name__=="__main__":
    run_REV()