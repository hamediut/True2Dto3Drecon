"""
Train our modified SliceGAN.
Before training you should run representative elementary size using RES.py and REV.py following the approach describe in our paper.
However, representative size should be too large such that your GPU memory is not enough.
In this case you can resize it to the closest size you can, you specify this with '' argument upon running the script.

Multi-GPU on Windows: uses DistributedDataParallel with gloo backend (NCCL is Linux-only).
Each GPU process handles its own portion of the batch, halving activation memory per GPU.
"""

import os
import sys
from glob import glob
import joblib
import numpy as np
import random
import argparse
import matplotlib.pyplot as plt
import tifffile

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from skimage.metrics import mean_squared_error

from src.training_utils import Dataset_3BSEs, calc_gradient_penalty, evaluate_G, Logger
from src.util_functions import _get_tensor_value, create_directories, plot_image_grid
from src.SMD_cal import calculate_two_point_list, list_to_df_two_point, calculate_two_point_3D

from src.networks import Generator, Discriminator

seed = 33


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
  parser.add_argument('--train_img_size', type= int, default= None,
                      help = 'training image size, it can be smaller than RES. if None, no resizing and RES will be used for training image size')
  parser.add_argument('--img_channels', type= int, default= 1,
                      help = 'number of channels. Default is 1 (binary image)')
  parser.add_argument('--z_channels', type = int, default = 16,
                      help = 'number of channles in noise vector.')
  parser.add_argument('--z_size', type =int, default = 4,
                      help ='size of noise vector. z dimension: (batch_size, z_channels, z_size, z_size, z_size)')
  parser.add_argument('--num_train_imgs', type = int, default = 320000,
                      help= 'Total number of training images from each 2D image for training')

  parser.add_argument('--batch_size', type = int, default = 2,
                      help= 'batch size per GPU for training the generator')
  parser.add_argument('--D_batch_size', type = int, default = 2, help= 'batch size per GPU for the discriminator')

  parser.add_argument('--ngpu', type = int, default= 1,
                      help= 'Number of GPUs to use. If > 1, uses DistributedDataParallel with gloo backend (works on Windows)')
  parser.add_argument('--lrg', type = float, default = 0.0001, help='learning rate for generator')
  parser.add_argument('--lrd', type = float, default = 0.0001, help='learning rate for discriminator')
  parser.add_argument('--Lambda', type = int, default = 10, help = 'lambda coefficient for gradient penalty')
  parser.add_argument('--critic_iters', type = int, default = 5, help = 'Number of times training D (critic) for one training of G')

  parser.add_argument('--save_every', type = int, default = 500, help = 'Define the frequency of model evaluation e.g., every 500 iterations')
  parser.add_argument('--mse_thresh', type = float, default = 5e-5, help= 'if mse btw s2_real and s2_fake is smaller than this, G and D will be saved')
  parser.add_argument('--num_img_eval', type =int, default= 8, help= 'Number of 3D images to generate during the training to calculate s2_fake and mse')
  parser.add_argument('--output_dir', type =str, help = 'Output directory to save models, images etc')
  parser.add_argument('--resume_nets', type = str, help= 'Path to the folder where the generator and discriminators to resume exist.')

  return parser.parse_args()


def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # gloo backend: supports GPU tensors on Windows (NCCL is Linux-only)
    dist.init_process_group(backend='gloo', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp():
    dist.destroy_process_group()


def train_worker(rank, world_size, args):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # deterministic=True restricts cuDNN to fewer algorithms and causes
        # "no valid cuDNN algorithm" OOM errors on large convolutions with DDP
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    random.seed(seed)
    np.random.seed(seed)

    use_ddp = world_size > 1
    is_main = (rank == 0)

    if use_ddp:
        setup_ddp(rank, world_size)

    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() and args.ngpu > 0 else 'cpu')

    training_data_path = [arg for arg in [args.dir_img_1, args.dir_img_2, args.dir_img_3] if arg is not None]

    if len(training_data_path) == 1:
        training_data_path *= 3
        isotropic = True
    else:
        isotropic = False

    training_params = {
        'num_Ds': len(training_data_path), 'batch_size': args.batch_size, 'D_batch_size': args.D_batch_size,
        'lrg': args.lrg, 'lrd': args.lrd, 'Lambda': args.Lambda, 'critic_iters': args.critic_iters,
        'z_size': args.z_size, 'train_img_size': args.train_img_size, 'RES': args.RES, 'resume_path': args.resume_nets
    }

    # All setup/logging/file I/O only on rank 0
    if is_main:
        print(f'training data path:{training_data_path}')
        print('--------------------------------')
        print(f'Training parameters: {training_params}')
        print('--------------------------------')
        output_folder = args.output_dir if args.output_dir else os.getcwd()
        current_run_folder, checkpoints_folder, best_folder, plots_imgs_folder = create_directories(output_folder, training_params=training_params)
        joblib.dump(training_params, os.path.join(current_run_folder, 'training_params.pkl'))
        Logger(file_name=os.path.join(current_run_folder, 'log.txt'), file_mode='a', should_flush=True)
        print(f'Creating output folders. Running folder: {current_run_folder}')
        print('--------------------------------')
        if isotropic:
            print(f'One 2D image provided --> isotropic microstructure.')
        else:
            print(f'{len(training_data_path)} two-dimensional images provided--> anisotropic microstructures.')

    ### loading dataset
    resized_to = None if args.train_img_size == args.RES else args.train_img_size
    if is_main:
        print(f'resized_to is:{resized_to}')

    ds1 = Dataset_3BSEs(args.dir_img_1, args.dir_img_2, args.dir_img_3,
                        patch_size=args.RES, resized_to=resized_to, num_samples=args.num_train_imgs)

    if use_ddp:
        sampler = DistributedSampler(ds1, num_replicas=world_size, rank=rank, shuffle=True)
        dataloader = DataLoader(ds1, batch_size=args.batch_size, sampler=sampler)
    else:
        dataloader = DataLoader(ds1, batch_size=args.batch_size, shuffle=True)

    # s2 reference statistics: compute only on rank 0 to avoid duplicate work
    if is_main:
        samples = ds1.sample(batch_size=100, return_s2=None)
        batches = [_get_tensor_value(sample.squeeze(1)).astype(np.uint8) for sample in samples]

        s2_list_x, f2_list_x = calculate_two_point_list(batches[0])
        s2_list_y, f2_list_y = calculate_two_point_list(batches[1])
        s2_df_x, f2_df_x = list_to_df_two_point(s2_list_x, f2_list_x)
        s2_df_y, f2_df_y = list_to_df_two_point(s2_list_y, f2_list_y)

        if len(batches) == 3:
            s2_list_z, f2_list_z = calculate_two_point_list(batches[2])
            s2_df_z, f2_df_z = list_to_df_two_point(s2_list_z, f2_list_z)
            s2_real_avg = (s2_df_x['s2']['mean'] + s2_df_y['s2']['mean'] + s2_df_z['s2']['mean']) / 3
            f2_real_avg = (f2_df_x['f2']['mean'] + f2_df_y['f2']['mean'] + f2_df_z['f2']['mean']) / 3
        elif len(batches) == 2:
            s2_real_avg = (s2_df_x['s2']['mean'] + s2_df_y['s2']['mean']) / 2
            f2_real_avg = (f2_df_x['f2']['mean'] + f2_df_y['f2']['mean']) / 2

        plot_image_grid(batches[0][:16, :, :], nrows=4, ncols=4, title='x-direction', output_folder=current_run_folder, file_name='Reals_x')
        tifffile.imwrite(os.path.join(current_run_folder, 'Real_x.tif'), batches[0])
        if len(batches) == 2:
            plot_image_grid(batches[1][:16, :, :], nrows=4, ncols=4, title='y-direction', output_folder=current_run_folder, file_name='Reals_y')
            tifffile.imwrite(os.path.join(current_run_folder, 'Real_y.tif'), batches[1])
        elif len(batches) == 3:
            plot_image_grid(batches[1][:16, :, :], nrows=4, ncols=4, title='y-direction', output_folder=current_run_folder, file_name='Reals_y')
            plot_image_grid(batches[2][:16, :, :], nrows=4, ncols=4, title='z-direction', output_folder=current_run_folder, file_name='Reals_z')
            tifffile.imwrite(os.path.join(current_run_folder, 'Real_y.tif'), batches[1])
            tifffile.imwrite(os.path.join(current_run_folder, 'Real_z.tif'), batches[2])

        joblib.dump(s2_real_avg, os.path.join(current_run_folder, 's2_real_avg.pkl'))
        joblib.dump(f2_real_avg, os.path.join(current_run_folder, 'f2_real_avg.pkl'))

    # Network architecture
    lays = 10
    dk, gk = [4]*lays, [4]*lays
    d_strides, g_strides = [2]*lays, [2]*lays
    df, gf = [args.img_channels, 64, 128, 256, 512, 1024, 2048, 1], [args.z_channels, 2048, 1024, 512, 256, 128, 64, 64, args.img_channels]
    dp, gp = [1, 1, 1, 1, 1, 1, 1,], [2] * 8

    if is_main:
        print('-----------------------------')
        print('Checking cuda availability...')
        print('Active CUDA Device: GPU', torch.cuda.current_device())
        print('Available devices ', torch.cuda.device_count())
        print('Current cuda device ', torch.cuda.current_device())
        if use_ddp:
            print(f'Using DistributedDataParallel across {world_size} GPUs (gloo backend)')
        print('-----------------------------')

    netG = Generator(num_layers=lays, gf=gf, gk=gk, gs=g_strides, gp=gp).to(device)
    if use_ddp:
        # find_unused_parameters=True is required because Generator.__init__ creates bns[-1]
        # but forward() only uses bns[:-1], so the last BN's parameters never receive gradients.
        netG = DDP(netG, device_ids=[rank], find_unused_parameters=True)
    optG = optim.Adam(netG.parameters(), lr=args.lrg, betas=(0.5, 0.99))

    netDs = []
    optDs = []
    for i in range(len(training_data_path)):
        netD = Discriminator(num_layers=lays, df=df, dk=dk, ds=d_strides, dp=dp).to(device)
        if use_ddp:
            netD = DDP(netD, device_ids=[rank])
        netDs.append(netD)
        optDs.append(optim.Adam(netDs[i].parameters(), lr=args.lrd, betas=(0.5, 0.99)))

    if args.resume_nets:
        if is_main:
            print(f'Resume training from snapshots in:{args.resume_nets}')
        files = os.listdir(args.resume_nets)
        if files[0].startswith('WGAN') and files[0].endswith('.pt'):
            iter_num = files[0].split('_')[3].split('.')[0]

        def _load(net, path):
            state = torch.load(path, map_location=device)
            (net.module if use_ddp else net).load_state_dict(state)

        _load(netG,    os.path.join(args.resume_nets, f'WGAN_Gen_iter_{iter_num}.pt'))
        _load(netDs[0], os.path.join(args.resume_nets, f'WGAN_Disc0_iter_{iter_num}.pt'))
        _load(netDs[1], os.path.join(args.resume_nets, f'WGAN_Disc1_iter_{iter_num}.pt'))
        if len(training_data_path) == 3:
            _load(netDs[2], os.path.join(args.resume_nets, f'WGAN_Disc2_iter_{iter_num}.pt'))

    if is_main:
        print('training loop started...')

    losses_dict = {'gen': [], 'disc_loss_real': [], 'disc_loss_gen': []}
    mse_dict = {}
    min_mse = 1

    for i, data_batches in enumerate(dataloader, start=1):
        if use_ddp:
            sampler.set_epoch(i)

        netG.train()
        dataset = [d.to(device) for d in data_batches]
        len_dataset = len(dataset)

        noise = torch.randn(args.D_batch_size, args.z_channels, args.z_size, args.z_size, args.z_size, device=device)
        # Use the unwrapped module so DDP hooks don't fire during D training.
        # G is not being trained here (.detach() follows), so no gradient sync is needed.
        _netG_unwrapped = netG.module if use_ddp else netG
        with torch.no_grad():
            fake_data = _netG_unwrapped(noise)
        sum_D_loss_real = 0
        sum_D_loss_gen = 0

        d1s = [2, 3] if len(training_data_path) == 2 else [2, 3, 4]
        d2s = [3, 2] if len(training_data_path) == 2 else [3, 2, 2]
        d3s = [4, 4] if len(training_data_path) == 2 else [4, 4, 3]

        for dim, (netD, optimizer, data, d1, d2, d3) in enumerate(zip(netDs, optDs, dataset, d1s, d2s, d3s)):
            if isotropic:
                netD = netDs[0]
                optimizer = optDs[0]

            netD.zero_grad()
            real_data = data
            out_real = netD(real_data).view(-1).mean()

            fake_data_perm = fake_data.permute(0, d1, 1, d2, d3).reshape(
                args.train_img_size * args.D_batch_size, args.img_channels, args.train_img_size, args.train_img_size)
            out_fake = netD(fake_data_perm).mean()

            g_penalty = calc_gradient_penalty(netD, real_data, fake_data_perm[:args.batch_size],
                                               args.batch_size, args.train_img_size,
                                               device, args.Lambda, args.img_channels)
            disc_cost = out_fake - out_real + g_penalty

            sum_D_loss_real += _get_tensor_value(out_real + g_penalty)
            sum_D_loss_gen  += _get_tensor_value(out_fake)

            disc_cost.backward()
            optimizer.step()

        if is_main:
            joblib.dump(losses_dict, os.path.join(current_run_folder, 'disc_losses.pkl'))

        if i % args.critic_iters == 0:
            netG.zero_grad()
            errG = 0

            if torch.cuda.is_available():
                alloc = torch.cuda.memory_allocated(rank) / 1024**3
                reserv = torch.cuda.memory_reserved(rank) / 1024**3
                sys.__stdout__.write(f"[Rank {rank}] G step iter {i}: {alloc:.2f}GB alloc / {reserv:.2f}GB reserved\n")
                sys.__stdout__.flush()

            noise = torch.randn(args.batch_size, args.z_channels, args.z_size, args.z_size, args.z_size, device=device).contiguous()
            fake = netG(noise)

            for dim, (netD, d1, d2, d3) in enumerate(zip(netDs, d1s, d2s, d3s)):
                if isotropic:
                    netD = netDs[0]
                fake_data_perm = fake.permute(0, d1, 1, d2, d3).reshape(
                    args.train_img_size * args.batch_size, args.img_channels,
                    args.train_img_size, args.train_img_size)
                # Use unwrapped module so D's DDP hooks don't fire during G's backward.
                # Only G's gradient sync should happen here; syncing all 4 DDP models
                # simultaneously via gloo causes CUDA context corruption on Windows.
                d_forward = netD.module if use_ddp else netD
                output = d_forward(fake_data_perm)
                errG -= output.mean()

            try:
                errG.backward()
            except RuntimeError as e:
                sys.__stdout__.write(f"[Rank {rank}] ERROR in errG.backward() at iter {i}: {e}\n")
                sys.__stdout__.flush()
                raise
            optG.step()

            losses_dict['gen'].append(float(_get_tensor_value(errG)))
            losses_dict['disc_loss_real'].append(sum_D_loss_real / len_dataset)
            losses_dict['disc_loss_gen'].append(sum_D_loss_gen / len_dataset)

            if is_main:
                print(f"Iteration= {i} \t Gen_loss={losses_dict['gen'][-1]: .3f} \t D_loss_real={losses_dict['disc_loss_real'][-1]:.3f} \t D_loss_gen={losses_dict['disc_loss_gen'][-1]:.3f}")

        if (i % args.save_every == 0) and (i >= 50) and is_main:
            print(f"Evaluating the model...")
            random_stack = np.random.randint(0, args.num_img_eval)

            # Strip DDP wrapper for evaluation
            eval_netG = netG.module if use_ddp else netG
            fake_np_binary, s2_fake_x, s2_fake_y, s2_fake_z, s2_fake_3D_avg = evaluate_G(
                netG=eval_netG, num_img=args.num_img_eval, directional=True)

            mse_3d_avg = mean_squared_error(s2_real_avg, s2_fake_3D_avg['s2']['mean'])
            mse_dict[f'iter_{i}'] = mse_3d_avg
            joblib.dump(mse_dict, os.path.join(current_run_folder, 'mse_dict.pkl'))
            print(f" Mean Squared Error (MSE) = {mse_3d_avg}")

            plot_image_grid(fake_np_binary[random_stack, :16, :, :], figsize=(3, 3), title='Fake-x',
                            output_folder=plots_imgs_folder, file_name=f'fake_x_iter_{i}')
            plot_image_grid(np.transpose(fake_np_binary[random_stack, :, :16, :], (1, 0, 2)), title='Fake-Y',
                            output_folder=plots_imgs_folder, file_name=f'fake_y_iter_{i}')
            plot_image_grid(np.transpose(fake_np_binary[random_stack, :, :, :16], (2, 0, 1)), title='Fake-Z',
                            output_folder=plots_imgs_folder, file_name=f'fake_z_iter_{i}')

            plt.figure()
            plt.plot(s2_df_x.index, s2_df_x['s2']['mean'], color='b',  label='Real x')
            plt.plot(s2_df_y.index, s2_df_y['s2']['mean'], color='b', linestyle='--', label='Real y')
            plt.plot(s2_real_avg, color='g', label='Real avg')
            plt.plot(s2_fake_x.index, s2_fake_x['s2']['mean'], color='r', label='Fake x')
            plt.plot(s2_fake_y.index, s2_fake_y['s2']['mean'], color='r', linestyle='--', label='Fake y')
            plt.plot(s2_fake_z.index, s2_fake_z['s2']['mean'], color='r', linestyle=':', label='Fake z')
            plt.plot(s2_fake_3D_avg.index, s2_fake_3D_avg['s2']['mean'], color='k', label='Fake avg')
            plt.xlabel('r(px)', fontsize='x-large')
            plt.ylabel('$S_2$', fontsize='x-large')
            plt.legend(ncol=2, fontsize='large')
            plt.title(f'Iteration: {i}, MSE ={mse_3d_avg:.4e}')
            plt.savefig(os.path.join(plots_imgs_folder, f's2_iter_{i}.png'), dpi=300)
            plt.close()

            # Get state dicts (unwrap DDP if needed)
            save_G  = (netG.module if use_ddp else netG).state_dict()
            save_Ds = [(netDs[j].module if use_ddp else netDs[j]).state_dict() for j in range(len_dataset)]

            if (mse_3d_avg < args.mse_thresh) or i % 10000 == 0:
                torch.save(save_G, os.path.join(checkpoints_folder, f'WGAN_Gen_iter_{i}.pt'))
                for D_index in range(len_dataset):
                    torch.save(save_Ds[D_index], os.path.join(best_folder, f'WGAN_Disc{D_index}_iter_{i}.pt'))

                if mse_3d_avg < min_mse:
                    for filename in glob(os.path.join(best_folder, 'WGAN*')):
                        os.remove(filename)
                    torch.save(save_G, os.path.join(best_folder, f'WGAN_Gen_iter_{i}.pt'))
                    min_mse = mse_3d_avg

        if use_ddp:
            dist.barrier()

    if use_ddp:
        cleanup_ddp()


def train():
    args = parse_args()
    world_size = args.ngpu if (args.ngpu > 0 and torch.cuda.is_available()) else 1

    if world_size > 1:
        mp.spawn(train_worker, args=(world_size, args), nprocs=world_size, join=True)
    else:
        train_worker(0, 1, args)


if __name__ == '__main__':
    train()
