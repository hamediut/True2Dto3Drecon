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


from src.SMD_cal import REV


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

    np.random.seed(args.seed)

    original_img = tifffile.imread(args.image_dir).astype(np.uint8)
    # # check if it is binary: 0:solid, 1: pore
    ## run REV function on the list of image
    s2_3d_dict, f2_3d_dict = REV(original_img, img_size_list = args.image_sizes,
                                    n_rand_samples=args.n_rnd_samples)
    
    joblib.dump(s2_3d_dict, os.path.join(args.output_dir, 's2_3d_dict.pkl'))
    joblib.dump(f2_3d_dict, os.path.join(args.output_dir, 'f2_3d_dict.pkl'))
    


if __name__=="__main__":
    run_REV()