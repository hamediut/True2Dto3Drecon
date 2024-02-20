"""
Train our modified SliceGAN. 
Before training you should run representative elementary size using RES.py and REV.py following the approach describe in our paper.
However, representative size should be too large such that your GPU memory is not enough. 
In this case you can resize it to the closest size you can, you specify this with '' argument upon running the script. 

"""

import os
from glob import glob
import joblib 
import numpy as np
import random
import argparse
import matplotlib.pyplot as plt


import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim

from skimage.metrics import mean_squared_error

from src.training_utils import Dataset_3BSEs, calc_gradient_penalty, evaluate_G, Logger
from src.util_functions import _get_tensor_value, create_directories, plot_image_grid
from src.networks import Generator, Discriminator

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
  parser.add_argument('--dir_img_1', required= True, type=str,
                       help='Full path to the 2D image taken from plane 1. it should be .tif file')
  parser.add_argument('--dir_img_2', type=str,
                        help='Full path to the 2D image taken from plane 2. it should be .tif file')
  parser.add_argument('--dir_img_3', type=str,
                        help='Full path to 2D image taken on plane 3. it should be .tif file')
  parser.add_argument('--RES', required= True, type = int,
                      help = 'Representative image size' )
  parser.add_argument('--train_img_size', type =int, default= 256,
                      help = 'training image size, it can be smaller than RES.')
  parser.add_argument('--img_channels', type= int, default= 1,
                      help = 'number of channels. Default is 1 (binary image)')
  parser.add_argument('--z_channels', type = int, default = 16,
                      help = 'number of channles in noise vector.')
  parser.add_argument('--z_size', type =int, default = 4,
                      help ='size of noise vector. z dimension: (batch_size, z_channels, z_size, z_size, z_size)')
  parser.add_argument('--num_train_imgs', type =int, default= 320000,
                      help= 'Total number of training images from each 2D image for training')
  parser.add_argument('--batch_size', type = int, default= 2,
                      help= 'batch size for training the model. due to the gpu limitation we used 2. Use larger values if possible')
  parser.add_argument('--D_batch_size', type = int, default=2, help= 'batch size for D')
  parser.add_argument('--ngpu', type = int, default= 1,
                      help= 'Number of available gpus. if ngpu > 1, model will be parallelized on ngpus')
  parser.add_argument('--lrg', type = float, default = 0.0001, help='learning rate for generator')
  parser.add_argument('--lrd', type = float, default = 0.0001, help='learning rate for discriminator')
  parser.add_argument('--Lambda', type= int, default=10, help = 'lambda coefficient for gradient penalty')
  parser.add_argument('--critic_iters', type = int, default = 5, help = 'Number of times training D (critic) for one training of G')

  parser.add_argument('--save_every', type = int, default = 500, help = 'Define the frequency of model evaluation e.g., every 500 iterations')
  parser.add_argument('--mse_thresh', type = float, default = 5e-5, help= 'if mse btw s2_real and s2_fake is smaller than this, G and D will be saved')
  parser.add_argument('--num_img_eval', type =int, default= 8, help= 'Number of 3D images to generate during the training to calculate s2_fake and mse')
  parser.add_argument('--output_dir', type =str, help = 'Output directory to save models, images etc')

  
  return parser.parse_args()

def train():
   args = parse_args()
   

   training_data_path = [args.dir_img_1, args.dir_img_2, args.dir_img_3]

   if len(training_data_path)==1:
      training_data_path *= 3
      isotropic = True
   else:
      isotropic = False

   training_params = {
      'num_Ds': len(training_data_path), 'batch_size':args.batch_size,'D_batch_size':args.D_batch_size,
      'lrg':args.lrg, 'lrd':args.lrd, 'Lambda': args.Lambda, 'critic_iters': args.critic_iters,
      'z_size': args.z_size, 'train_img_size': args.train_img_size, 'RES': args.RES
      }


   output_folder = args.output_dir if args.output_dir else os.getcwd()
   
   current_run_folder, checkpoints_folder, best_folder, plots_imgs_folder = create_directories(output_folder, training_params=training_params)
   joblib.dump(training_params, os.path.join(current_run_folder, 'training_params.pkl'))

   Logger(file_name=os.path.join(current_run_folder, 'log.txt'), file_mode='a', should_flush=True)
   print('--------------------------------')
   print(f'Training parameters: {training_params}')
   print('--------------------------------')
   print(f'Creating output folders. Running folder: {current_run_folder}')

   print('--------------------------------')
   if isotropic:
      print(f'One 2D image provided --> isotropic microstructure.')
   else:
      print(f'{len(training_data_path)} two-dimensional images provided--> anisotropic microstructures.')



   ### loafing dataset----------------------------------------------
   ds1 = Dataset_3BSEs(args.dir_img_1, args.dir_img_2, args.dir_img_3,
                        patch_size = args.RES, resized_to = args.train_img_size,  num_samples = args.num_train_imgs)
   dataloader_xyz = DataLoader(ds1, batch_size=args.batch_size, shuffle=True)

   real_np_x, real_np_y, real_np_z = ds1.sample(batch_size= 100, return_s2= None)
   plot_image_grid(real_np_x[:16, :, :], nrows= 4, ncols=4, title= 'x-direction',output_folder=current_run_folder, file_name= 'Reals_x')
   plot_image_grid(real_np_y[:16, :, :], nrows= 4, ncols=4, title= 'y-direction',output_folder=current_run_folder, file_name= 'Reals_y')
   plot_image_grid(real_np_z[:16, :, :], nrows= 4, ncols=4, title= 'z-direction',output_folder=current_run_folder, file_name= 'Reals_z')

   _, _, _, s2_real_avg = ds1.sample(batch_size= 100, return_s2= True)
   joblib.dump(s2_real_avg, os.path.join(current_run_folder, 's2_real_avg.pkl'))
   
   # here we sample 100 images in each plane to calculate average s2.
   # Then, we compute mse between this average and average value of fakes images and use it as a criterion for saving best models.
   # print('Constructing networks...')
   # Layers in G and D
   lays =10
   # kernals for each layer
   dk, gk = [4]*lays, [4]*lays
   # strides
   ds, gs= [2]*lays, [2]*lays
   # with conv layer and sticking to rules to avoid checkerboard
   df, gf = [args.img_channels, 64, 128, 256, 512, 1024, 2048, 1], [args.z_channels, 2048, 1024, 512, 256, 128, 64, 64,  args.img_channels]
   dp, gp = [1, 1, 1, 1, 1, 1, 1,], [2] * 8 # for res = 256, sticking to rules for checkerboard

   print('-----------------------------')
   print('Checking cuda availability...')
   cuda = torch.cuda.is_available()
   print('Active CUDA Device: GPU', torch.cuda.current_device())
   print ('Available devices ', torch.cuda.device_count())
   print ('Current cuda device ', torch.cuda.current_device())
   device = torch.device("cuda:0" if(torch.cuda.is_available() and args.ngpu > 0) else "cpu")
   # print('-----------------------------')
   
   netG = Generator(num_layers= lays, gf=gf, gk=gk, gs=gs, gp=gp).to(device)
   if ('cuda' in str(device)) and (args.ngpu >1):
      netG =nn.DataParallel(netG, list(range(args.ngpu)))
   optG = optim.Adam(netG.parameters(), lr=args.lrg, betas=(0.5, 0.99))

   # Define 1 Discriminator and optimizer for each plane in each dimension
   netDs = []
   optDs = []

   for i in range(len(training_data_path)):
      netD = Discriminator(num_layers=lays, df=df, dk= dk, ds=ds, dp=dp).to(device)

      if ('cuda' in str(device)) and (args.ngpu > 1):
         netD = (nn.DataParallel(netD, list(range(args.gpu)))).to(device)
      netDs.append(netD)
      optDs.append(optim.Adam(netDs[i].parameters(), lr= args.lrd, betas= (0.5, 0.99)))

   print('-----------------------------')
   print('training loop started...')

   losses_dict = {}
   losses_dict['gen'] = []
   losses_dict['3disc_loss_real'] = []
   losses_dict['3disc_loss_gen'] = []

   mse_dict= {}
   min_mse = 1 # we start with a large error and update it with the minimum mse obtained

   for i, (datax, datay, dataz) in enumerate(dataloader_xyz, start = 1):
      netG.train()
      dataset = [datax, datay, dataz] # this is the first, second, and third images in your training dataset
      len_dataset = len(dataset) # = 3

      noise = torch.randn(args.D_batch_size, args.z_channels, args.z_size, args.z_size, args.z_size, device = device)
      fake_data = netG(noise).detach()
      sum_D_loss_real = 0
      sum_D_loss_gen = 0

      for dim, (netD, optimizer, data, d1, d2, d3) in enumerate(zip(netDs, optDs, dataset, [2, 3, 4], [3, 2, 2], [4, 4, 3])):
              
          if isotropic:
              netD = netDs[0]
              optimizer = optDs[0]

          netD.zero_grad()

          real_data = data.to(device)
          out_real = netD(real_data).view(-1).mean()

          fake_data_perm = fake_data.permute(0, d1, 1, d2, d3).reshape(args.train_img_size * args.D_batch_size,
                                                                        args.img_channels, args.train_img_size,
                                                                          args.train_img_size)
          out_fake = netD(fake_data_perm).mean()

          # compute gradient penalty
          g_penalty = calc_gradient_penalty(netD, real_data, fake_data_perm[:args.batch_size],
                                             args.batch_size, args.train_img_size,
                                             device, args.Lambda, args.img_channels)
          disc_cost = out_fake - out_real + g_penalty

          sum_D_loss_real += _get_tensor_value(out_real + g_penalty)
          sum_D_loss_gen  += _get_tensor_value(out_fake)

          disc_cost.backward()
          optimizer.step()

      joblib.dump(losses_dict, os.path.join(current_run_folder, 'disc_losses.pkl'))

      # training G
      if i % args.critic_iters ==0:
         
         
         netG.zero_grad()
         errG = 0
         noise = torch.randn(args.batch_size, args.z_channels, args.z_size, args.z_size, args.z_size, device=device).contiguous()
         fake = netG(noise) #(batch_size, img_channels, train_img_size, train_img_size, train_img_size)
         
         for dim, (netD, d1, d2, d3) in enumerate(zip(netDs, [2, 3, 4], [3, 2, 2], [4, 4, 3])):

            if isotropic:
                #only need one D
                netD = netDs[0]
            # permute and reshape to feed to disc
            fake_data_perm = fake.permute(0, d1, 1, d2, d3).reshape(args.train_img_size * args.batch_size, args.img_channels, 
                                                                    args.train_img_size, args.train_img_size)
            #fake_data_perm: (32, 64, 2, 64, 64) --> reshape to : (32 *64 = 2048, 2, 64, 64)
            output = netD(fake_data_perm)
            errG -= output.mean()

         errG.backward()
         optG.step()

         losses_dict['gen'].append(float(_get_tensor_value(errG)))
         losses_dict['3disc_loss_real'].append(sum_D_loss_real/3)
         losses_dict['3disc_loss_gen'].append(sum_D_loss_gen/3)
         
         print(f"Iteration= {i} \t Gen_loss={losses_dict['gen'][-1]: .3f} \t D_loss_real={losses_dict['3disc_loss_real'][-1]:.3f} \t D_loss_gen={losses_dict['3disc_loss_gen'][-1]:.3f}")
      if (i % args.save_every == 0)  and (i >= 50):
         print(f"Evaluating the model...")
         random_stack = np.random.randint(0 , args.num_img_eval)

        #  fake_np_binary, s2_fake_X, s2_fake_Y, s2_fake_Z, s2_fake_3D_avg = evaluate_G(netG= netG, num_img=args.num_img_eval)
         fake_np_binary, s2_fake_3D_avg, _ = evaluate_G(netG= netG, num_img=args.num_img_eval)
         
         mse_3d_avg = mean_squared_error(s2_real_avg, s2_fake_3D_avg['s2']['mean'])
         mse_dict[f'iter_{i}'] = mse_3d_avg
         joblib.dump(mse_dict, os.path.join(current_run_folder, 'mse_dict.pkl'))
         print(f" Mean Squared Error (MSE) = {mse_3d_avg}")
         ## saving fakes images and s2 plots----------------------------
         plot_image_grid(fake_np_binary[random_stack, :16, :, :], figsize=(3, 3), title= 'Fake-x',
                         output_folder= plots_imgs_folder, file_name= f'fake_x_iter_{i}' )
         plot_image_grid(np.transpose(fake_np_binary[random_stack, :, :16, :], (1, 0, 2)), title= 'Fake-Y',
                         output_folder=plots_imgs_folder, file_name=f'fake_y_iter_{i}')
         
         plot_image_grid(np.transpose(fake_np_binary[random_stack, :, :, :16], (2, 0, 1)), title= 'Fake-Z',
                         output_folder= plots_imgs_folder, file_name= f'fake_z_iter_{i}')
         plt.figure()
         plt.plot(s2_real_avg, color = 'k', label = 'Real Avg')
         plt.plot(s2_fake_3D_avg.index, s2_fake_3D_avg['s2']['mean'], color ='b', label='Fake Avg')
         plt.xlabel('r(px)', fontsize = 'x-large')
         plt.ylabel('$S_2$', fontsize = 'x-large')
         plt.legend(fontsize ='large')
         plt.title(f'Iteration: {i}, MSE ={mse_3d_avg:.4e}')
         plt.savefig(os.path.join(plots_imgs_folder, f's2_iter_{i}.png'), dpi = 300)
         
         ## saving models---------------------------------
         
         if (mse_3d_avg < args.mse_thresh) or i % 10000 == 0:
            
            torch.save(netG.state_dict(), os.path.join(checkpoints_folder, f'WGAN_Gen_iter_{i}.pt'))
#             torch.save(netD.state_dict(), os.path.join(output_checkpoints, f'WGAN_Disc_iter_{i}.pt'))
            if mse_3d_avg < min_mse:
                #removing the previous best model in the folder cause they're not the best anymore
                for filename in glob(os.path.join(best_folder, 'WGAN*')):
                    os.remove(filename) 
                # save the checkpoint as the best model
                torch.save(netG.state_dict(), os.path.join(best_folder, f'WGAN_Gen_iter_{i}.pt'))
                torch.save(netDs[0].state_dict(), os.path.join(best_folder, f'WGAN_Disc0_iter_{i}.pt'))
                if not isotropic: #if anisotropic, save all the discriminators for potentially resuming the training
                   torch.save(netDs[1].state_dict(), os.path.join(best_folder, f'WGAN_Disc1_iter_{i}.pt'))
                   torch.save(netDs[2].state_dict(), os.path.join(best_folder, f'WGAN_Disc2_iter_{i}.pt'))

                # updating the minimum mse 
                min_mse = mse_3d_avg


if __name__== "__main__":
    train() 

