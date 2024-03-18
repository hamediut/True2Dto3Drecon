"""
Functions and classes used in training.
"""

import torch
from torch import autograd
import numpy as np
from torch.utils.data import Dataset, DataLoader
# from torchvision.transforms import ToTensor
import tifffile
from skimage.filters import threshold_otsu
import PIL

import sys

from typing import Any, List, Tuple, Union, Optional

from src.util_functions import _get_tensor_value
from src.SMD_cal import calculate_two_point_list, list_to_df_two_point, calculate_two_point_3D


class Dataset_3BSEs(Dataset):
    def __init__(self, image1_path: str,
                  image2_path: str,
                  image3_path: Optional[str]= None,
                  patch_size: int =512,
                  resized_to: Optional[int] = None,
                  num_samples: int = 15000):
        self.image1_path = image1_path
        self.image2_path = image2_path
        self.image3_path = image3_path
        self.patch_size = patch_size
        self.num_samples = num_samples
        self.resized_to = resized_to
        #reading the original large images
        # if image is not zero and 1, we put the maximum value =1--> important for s2 calculation
        self.image1 = np.where(tifffile.imread(self.image1_path) >= 1, 1, 0).astype(np.uint8)
        self.image2 = np.where(tifffile.imread(self.image2_path) >= 1, 1, 0).astype(np.uint8)
        self.image3 = np.where(tifffile.imread(self.image3_path) >= 1, 1, 0).astype(np.uint8) if image3_path is not None else None
        
    def __len__(self):
        return self.num_samples  # Set the number of patches you want to extract

    def __getitem__(self, index):
        # Random crop coordinates for image1 and image2
        x1 = np.random.randint(0, self.image1.shape[0] - self.patch_size + 1)
        y1 = np.random.randint(0, self.image1.shape[1] - self.patch_size + 1)
        
        x2 = np.random.randint(0, self.image2.shape[0] - self.patch_size + 1)
        y2 = np.random.randint(0, self.image2.shape[1] - self.patch_size + 1)
        
        # Extract patches for image1 and image2
        patch1 = self.image1[x1:x1 + self.patch_size, y1:y1 + self.patch_size]
        patch2 = self.image2[x2:x2 + self.patch_size, y2:y2 + self.patch_size]

        patches = [patch1, patch2]

        if self.resized_to:

            resized_pathces = []

            for patch in patches:
                patch_pil = PIL.Image.fromarray(patch)
                patch_resized = patch_pil.resize((self.resized_to, self.resized_to), PIL.Image.LANCZOS)
                thresh = threshold_otsu(np.array(patch_resized))
                resized_patch = np.where(np.array(patch_resized) > thresh, 1, 0).astype(np.uint8)
                resized_pathces.append(resized_patch)
            patches = resized_pathces
        # Convert patches to PyTorch tensors
        patches = [torch.from_numpy(patch).unsqueeze(dim=0).type(torch.cuda.FloatTensor) for patch in patches]

        # If the third image is loaded, process it similarly
        if self.image3 is not None:
            x3 = np.random.randint(0, self.image3.shape[0] - self.patch_size + 1)
            y3 = np.random.randint(0, self.image3.shape[1] - self.patch_size + 1)
            patch3 = self.image3[x3:x3 + self.patch_size, y3:y3 + self.patch_size]

            if self.resized_to:
                patch3_pil = PIL.Image.fromarray(patch3)
                patch3_resized =  patch3_pil.resize((self.resized_to, self.resized_to), PIL.Image.LANCZOS)
                thresh = threshold_otsu(np.array(patch3_resized))
                patch3 = np.where(np.array(patch3_resized) > thresh, 1, 0).astype(np.uint8)
            patch3 = torch.from_numpy(patch3).unsqueeze(dim=0).type(torch.cuda.FloatTensor)
            patches.append(patch3)

        return tuple(patches)

    def sample(self, batch_size, return_s2 =None):
        """
        This method takes batch_size number of random images from each plane.
        If return_s2 = None, it only return the random images as tensors.
        If return_s2 =True, it calculates s2 in each plane and return them along with the average value
        """
        dataloader = DataLoader(self, batch_size= batch_size, shuffle = True)
        
        batches = next(iter(dataloader))
        return batches
        real_np_x = _get_tensor_value(batch_x)[:, 0, :, :].astype(np.uint8)
        real_np_y = _get_tensor_value(batch_y)[:, 0, :, :].astype(np.uint8)
        real_np_z = _get_tensor_value(batch_z)[:, 0, :, :].astype(np.uint8)

        s2_real_X_list, f2_real_X_list = calculate_two_point_list(real_np_x)
        s2_real_Y_list, f2_real_Y_list = calculate_two_point_list(real_np_y)
        s2_real_Z_list, f2_real_Z_list = calculate_two_point_list(real_np_z)

        s2_real_X, _ = list_to_df_two_point(s2_real_X_list, f2_real_X_list)
        s2_real_Y, _ = list_to_df_two_point(s2_real_Y_list, f2_real_Y_list)
        s2_real_Z, _ = list_to_df_two_point(s2_real_Z_list, f2_real_Z_list)

        s2_real_avg = (s2_real_X['s2']['mean'] + s2_real_Y['s2']['mean'] + s2_real_Y['s2']['mean'])/3

        if not return_s2:

            return real_np_x, real_np_y, real_np_z
        else:
            return s2_real_X, s2_real_Y, s2_real_Z, s2_real_avg

            
        
###--------------------------------------------------------------------------------------
def evaluate_G(netG, num_img, img_size = 256, z_size = 4, z_channels = 16, device = 'cuda'):

    
    netG.eval()
    with torch.no_grad():
        fake_np_stack_binary = np.zeros((num_img, img_size, img_size, img_size)).astype(np.uint8)

        for i in range(num_img):
            noise = torch.randn(1, z_channels, z_size, z_size, z_size, device=device)
            fake_np =_get_tensor_value(netG(noise))[0, 0, :, :, :]
            thresh = threshold_otsu(fake_np)
            fake_np_binary = np.where(fake_np> thresh, 1, 0).astype(np.uint8)
            fake_np_stack_binary[i] = fake_np_binary

        # s2_fake_X, s2_fake_Y, s2_fake_Z, s2_fake_3D_avg = calculate_two_point_3D(fake_np_stack_binary, directional = True) # average s2 in 3D
        s2_avg, f2_avg = calculate_two_point_3D(fake_np_stack_binary, directional= False)    
    
    netG.train()
    # return fake_np_stack_binary, s2_fake_X, s2_fake_Y, s2_fake_Z, s2_fake_3D_avg
    return fake_np_stack_binary, s2_avg, f2_avg

def calc_gradient_penalty(netD, real_data, fake_data, batch_size, img_size, device, Lambda, img_channels):
    """
    calculate gradient penalty for a batch of real and fake data
    :param netD: Discriminator network
    :param real_data:
    :param fake_data:
    :param batch_size:
    :param img_size: image size
    :param device:
    :param Lambda: coefficient for gradient penalty
    :param img_channels: image channels
    :return: gradient penalty
    """
    #sample and reshape random numbers
    alpha = torch.rand(batch_size, 1, device = device)
    alpha = alpha.expand(batch_size, int(real_data.nelement() / batch_size)).contiguous()
    alpha = alpha.view(batch_size, img_channels, img_size, img_size)

    # create interpolate dataset
    interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())
    interpolates.requires_grad_(True)

    #pass interpolates through netD
    disc_interpolates = netD(interpolates)
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size(), device = device),
                              create_graph=True, only_inputs=True)[0]
    # extract the grads and calculate gp
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * Lambda
    return gradient_penalty

class Logger(object):
    """Redirect stderr to stdout, optionally print stdout to a file, and optionally force flushing on both stdout and the file."""

    def __init__(self, file_name: str = None, file_mode: str = "w", should_flush: bool = True):
        self.file = None

        if file_name is not None:
            self.file = open(file_name, file_mode)

        self.should_flush = should_flush
        self.stdout = sys.stdout
        self.stderr = sys.stderr

        sys.stdout = self
        sys.stderr = self

    def __enter__(self) -> "Logger":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    def write(self, text: Union[str, bytes]) -> None:
        """Write text to stdout (and a file) and optionally flush."""
        if isinstance(text, bytes):
            text = text.decode()
        if len(text) == 0: # workaround for a bug in VSCode debugger: sys.stdout.write(''); sys.stdout.flush() => crash
            return

        if self.file is not None:
            self.file.write(text)

        self.stdout.write(text)
        # sys.__stdout__.write(text) # h: added to print to original stdout

        if self.should_flush:
            self.flush()

    def flush(self) -> None:
        """Flush written text to both stdout and a file, if open."""
        if self.file is not None:
            self.file.flush()

        self.stdout.flush()

    def close(self) -> None:
        """Flush, close possible files, and remove stdout/stderr mirroring."""
        self.flush()

        # if using multiple loggers, prevent closing in wrong order
        if sys.stdout is self:
            sys.stdout = self.stdout
        if sys.stderr is self:
            sys.stderr = self.stderr

        if self.file is not None:
            self.file.close()
            self.file = None