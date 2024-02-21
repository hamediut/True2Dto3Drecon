''''
This script runs representative elementary size (RES) analysis on a binary 2D image.

it takes path to the 2D tif file and a list of image sizes, and number of random samples as inputs to caluclate:
two-point correlation function (S2) and its scaled version  (F2) in 2D.
These values can then be analyzed for RES calculation for that image. 

'''

import os
import argparse
import numpy as np
import pandas as pd
import tifffile
# from numba import jit
# from tqdm import tqdm 
# from typing import List, Dict
import joblib


from src.SMD_cal import RES


def parse_args():
  """Parses arguments."""
  parser = argparse.ArgumentParser()
  parser.add_argument('--image_dir', required= True, type=str,
                       help='Full path to the large image file. it should be .tif file')
  parser.add_argument('--image_sizes', required =True, type=int, nargs='+',
                      help='List of the image sizes you want to do REV analysis with. provided with space: 64 128 256 etc')
  parser.add_argument('--n_rnd_samples', type=int, default=50,
                      help='Number of random subvolume of each size to calculate average S2, F2')
  parser.add_argument('--seed', type= int, default= 33)
  parser.add_argument('--output_dir', type=str,
                      help='Directory to save the results')
  return parser.parse_args()

def run_RES()-> None:
  args = parse_args()

  np.random.seed(args.seed)
  # read the original large image
  original_img = tifffile.imread(args.image_dir).astype(np.uint8)
  
  # check if it is binary: 0:solid, 1: pore
  if original_img.max() > 1:
        original_img = np.where(original_img > 1,1,0)
    
  # run RES function on the list of images
  s2_dict, f2_dict = RES(original_img, img_size_list= args.image_sizes,
                         n_rand_samples= args.n_rnd_samples)
  #here you can also caluclate s2 and f2 for your whole image.
  #but if it is too large it is gonna take time!

  joblib.dump(s2_dict, os.path.join(args.output_dir, 's2_dict.pkl'))
  joblib.dump(f2_dict, os.path.join(args.output_dir, 'f2_dict.pkl'))



if __name__=="__main__":
   run_RES()
        


  


