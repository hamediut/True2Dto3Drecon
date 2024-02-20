""""
In this script you can generate images using trained models. Run this after training models
"""

import argparse
import os
import numpy as np
import random
import tifffile
from skimage.filters import threshold_otsu
from skimage.morphology import closing, ball
from tqdm import tqdm
import torch
from torch import nn

from src.networks import Generator
from src.util_functions import _get_tensor_value


seed =33
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU.
    # Configure PyTorch to be deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
random.seed(seed)
np.random.seed(seed)


## parsing arguments
def parse_args():
  """Parses arguments."""
  parser = argparse.ArgumentParser()
  
  parser.add_argument('--G_pkl', required= True, help= "path to the trained generator .pt file")
  parser.add_argument('--ngpu', type = int, default= 1,
                      help= 'Number of available gpus. if ngpu > 1, model will be parallelized on ngpus')
  parser.add_argument('--output_dir', type = str, help= 'where to save generated images')
  parser.add_argument('--num_imgs', type =int, default= 50, help= 'Number of images to generate')
  parser.add_argument('--img_size', type = int, default = 256, help= 'Size of the generated 3D images: (img_size, img_size, img_size). if different than 256, you should change the networks architecture ')
  parser.add_argument('--z_size', type = int, default= 4, help= 'Size of noise vector. with z_size = 4, output image size = 256. with z_size = 6, output image size = 512.')
  parser.add_argument('--img_channels', type= int, default= 1,
                      help = 'number of channels. Default is 1 (binary image)')
  parser.add_argument('--z_channels', type = int, default = 16,
                      help = 'number of channles in noise vector.')
  parser.add_argument('--radius', type= int, default = 0, help= 'if > 0, the generated images will be post-processed with a spherical structuring element of specificed radius in pixels ')

  return parser.parse_args()



def gen_img():
   args = parse_args()

   print('-----------------------------')
   print('Checking cuda availability...')
#    cuda = torch.cuda.is_available()
   print('Active CUDA Device: GPU', torch.cuda.current_device())
   print ('Available devices ', torch.cuda.device_count())
   print ('Current cuda device ', torch.cuda.current_device())
   device = torch.device("cuda:0" if(torch.cuda.is_available() and args.ngpu > 0) else "cpu")

   print('-----------------------------')
   # Layers in G and D
   lays =10
   # kernals for each layer
   gk = [4]*lays
   # strides
   gs= [2]*lays
   # with conv layer and sticking to rules to avoid checkerboard
   gf = [args.z_channels, 2048, 1024, 512, 256, 128, 64, 64,  args.img_channels]
   gp = [2] * 8 # for res = 256, sticking to rules for checkerboard

   netG = Generator(num_layers= lays, gf=gf, gk=gk, gs=gs, gp=gp).to(device)
   if ('cuda' in str(device)) and (args.ngpu >1):
      netG =nn.DataParallel(netG, list(range(args.ngpu)))

   netG.load_state_dict(torch.load(args.G_pkl))
   netG.eval()

   print('Creating output folder...')
   if args.output_dir:
      output_folder = os.path.join(args.output_dir, f'inference_z_size{args.z_size}_ball_{args.radius}')
   else:
      output_folder = os.path.join(os.getcwd(), f'inference_z_size{args.z_size}_ball_{args.radius}')
#    output_folder = args.output_dir if args.output_dir else os.getcwd()
   print(f'Generated images will be saved in: {output_folder}')
   if not os.path.exists(output_folder):
     os.makedirs(output_folder)
   
   print('-----------------------------')

   print('Generate images...')

#    if args.radius > 0:
#       selem = ball(radius= args.radius)
   
   for i in tqdm(range(args.num_imgs)):
      num_zero = len(str(args.num_imgs)) - len(str(i))
      num = num_zero * str(0) + str(i)
      noise = torch.randn(1, args.z_channels, args.z_size, args.z_size, args.z_size, device=device)
      fake_np = _get_tensor_value(netG(noise))[0, 0, :, :, :]
      thresh = threshold_otsu(fake_np)
      fake_np_binary = np.where(fake_np > thresh, 1, 0).astype(np.uint8)

      if args.radius > 0:
         fake_np_binary = closing(fake_np_binary, ball(radius= args.radius))

      tifffile.imwrite(os.path.join(output_folder, f'fake3D_{num}.tif'), fake_np_binary)
   

if __name__ == '__main__':
   gen_img()




