## Here you can find the functions to compute different correlations functions.

import os
import numpy as np
import pandas as pd
import tifffile
from tqdm import tqdm
from typing import List, Dict
from numba import jit

## correlation functions for 2D images-------------------------
@jit 
# --> It is preferred to use numba here for a speed-up, if installed!!
def two_point_correlation(im, dim, var=1):
    """
    This method computes the two point correlation,
    also known as second order moment,
    for a segmented binary image in the three principal directions.
    
    dim = 0: x-direction
    dim = 1: y-direction
    dim = 2: z-direction
    
    var should be set to the pixel value of the pore-space. (Default 1)
    
    The input image im is expected to be two-dimensional.
    """
    if dim == 0: #x_direction
        dim_1 = im.shape[1] #y-axis
        dim_2 = im.shape[0] #x-axis
    elif dim == 1: #y-direction
        dim_1 = im.shape[0] #x-axis
        dim_2 = im.shape[1] #z-axis
        
    two_point = np.zeros((dim_1, dim_2))
    for n1 in range(dim_1):
        for r in range(dim_2):
            lmax = dim_2-r
            for a in range(lmax):
                if dim == 0:
                    pixel1 = im[a, n1]
                    pixel2 = im[a+r, n1]
                elif dim == 1:
                    pixel1 = im[n1, a]
                    pixel2 = im[n1, a+r]

                if pixel1 == var and pixel2 == var:
                    two_point[n1, r] += 1
            two_point[n1, r] = two_point[n1, r]/(float(lmax))
    return two_point

def calculate_two_point_df(images):
    """
    This function calculates average two-point correlations (s2 and fn) from images and convert them to dataframe.
    """
    
    s2_list = []
    fn_list = []
    
    for i in range(images.shape[0]):
        # 1) convert each image in the batch to microstructure
        # 2) calculate the requested polytope function including scaled version
        # 3) append the results to the empty list above
        
        two_pt_dim0 = two_point_correlation(images[i], dim = 0, var = 1) #S2 in x-direction
        two_pt_dim1 = two_point_correlation(images[i], dim = 1, var = 1) #S2 in y-direction

        #Take average of directions; use half linear size assuming equal dimension sizes
        Nr = two_pt_dim0.shape[0]//2

        S2_x = np.average(two_pt_dim1, axis=0)[:Nr]
        S2_y = np.average(two_pt_dim0, axis=0)[:Nr]
        S2_average = ((S2_x + S2_y)/2)[:Nr]
        
        s2_list.append(S2_average)
        
        # autoscaled covriance---------------------------------------
        # f_average = (S2_average - S2_average[0]**2)/S2_average[0]/(1 - S2_average[0])
        f_average = cal_fn(S2_average, n= 2)
        fn_list.append(f_average)
    
    # from list to dataframe----------
    
    df_list = []
    for i in np.arange(0, len(s2_list)):
        df_list.append(pd.DataFrame(s2_list[i], columns = ['s2'] ) )
    df = pd.concat(df_list)
    df['r'] = df.index
    df_grouped = df.groupby( ['r'] ).agg( {'s2': [np.mean, np.std, np.size] } )
    
    
    df_fn_list = []
    for i in np.arange(0, len(fn_list)):
        df_fn_list.append(pd.DataFrame(fn_list[i], columns = ['fn'] ) )
    df_fn = pd.concat(df_fn_list)
    df_fn['r'] = df_fn.index
    df_fn_grouped = df_fn.groupby( ['r'] ).agg( {'fn': [np.mean, np.std, np.size] } )
        
        
    return df_grouped, df_fn_grouped


@jit
def two_point_correlation3D(im, dim, var=1):
    """
    This method computes the two point correlation,
    also known as second order moment,
    for a segmented binary image in the three principal directions.
    
    dim = 0: x-direction
    dim = 1: y-direction
    dim = 2: z-direction
    
    var should be set to the pixel value of the pore-space. (Default 1)
    
    The input image im is expected to be three-dimensional.
    """
    if dim == 0: #x_direction
        dim_1 = im.shape[2] #y-axis
        dim_2 = im.shape[1] #z-axis
        dim_3 = im.shape[0] #x-axis
    elif dim == 1: #y-direction
        dim_1 = im.shape[0] #x-axis
        dim_2 = im.shape[1] #z-axis
        dim_3 = im.shape[2] #y-axis
    elif dim == 2: #z-direction
        dim_1 = im.shape[0] #x-axis
        dim_2 = im.shape[2] #y-axis
        dim_3 = im.shape[1] #z-axis
        
    two_point = np.zeros((dim_1, dim_2, dim_3))
    for n1 in range(dim_1):
        for n2 in range(dim_2):
            for r in range(dim_3):
                lmax = dim_3-r
                for a in range(lmax):
                    if dim == 0:
                        pixel1 = im[a, n2, n1]
                        pixel2 = im[a+r, n2, n1]
                    elif dim == 1:
                        pixel1 = im[n1, n2, a]
                        pixel2 = im[n1, n2, a+r]
                    elif dim == 2:
                        pixel1 = im[n1, a, n2]
                        pixel2 = im[n1, a+r, n2]
                    
                    if pixel1 == var and pixel2 == var:
                        two_point[n1, n2, r] += 1
                two_point[n1, n2, r] = two_point[n1, n2, r]/(float(lmax))
    return two_point


def cal_fn( polytope, n):
    """This function calculates scaled autocovariance function from Pn function.
    polytope:polytope function it can be two point correlation function (s2) or 
    higher order functions such as p3, p4, etc
    n: order of polytope e.g., n =2 for two-point correlation (s_2), n= 3 for p3h and p3v"""
    numerator = polytope - polytope[0] ** n
    denominator = polytope[0] - polytope[0] ** n
    fn_r = numerator/ denominator
    return fn_r

def calculate_two_point_3D(images, directional = None):
    """
    This function calculates average two-point correlation in 3D.
    Inputs: 
    Can be a 3D image (numpy array)
    or 
    4D array (couple of 3D images) whose first dimension is the number of images: (number_imgs, image_size, image_size, image_size).
    
    Returns:
    in the case of 3D image:
        --> it returns 4 numpy consisting two-point correlation in x, y, and z direction plus the average of them.

    in case of 4D images:
    --> it calculates S_2 and scaled-autocovariance (F_2) for all images,
    and return dataframes containing average s_2 and f_2, plus std and number of samples for each r.

    Note that correlation function is calculated from r = 0 to  half of the smallest dimension in the image.
    For instance if the image shape is (512, 256, 256), s2 size is 128.

    """
    
#     print(len(images.shape))
    if len(images.shape) == 3:
        Nr = min(images.shape)//2
        # only 1 3D image
        
        two_point_covariance = {}
        for j, direc in tqdm(enumerate( ["x", "y", "z"]) ):
            two_point_direc =  two_point_correlation3D(images, dim = j, var = 1)
            two_point_covariance[direc] = two_point_direc
        direc_covariances = {}

        for direc in ["x", "y", "z"]:
            direc_covariances[direc] =  np.mean(np.mean(two_point_covariance[direc], axis=0), axis=0)[: Nr]
        average = (np.array(direc_covariances['x']) + np.array(direc_covariances['y']) + np.array(direc_covariances['z']) )/3
        
        return np.array(direc_covariances['x']), np.array(direc_covariances['y']), np.array(direc_covariances['z']), average
    
    elif len(images.shape) == 4 and not directional:
        Nr = min(images.shape[1:])//2
        s2_list = []
        f2_list = []

        for i in range(images.shape[0]):


            two_point_covariance = {}
            for j, direc in enumerate(["x", "y", "z"]) :
                two_point_direc = two_point_correlation3D(images[i], j, var = 1)
                two_point_covariance[direc] = two_point_direc
        
            direc_covariances = {}
            for direc in ["x", "y", "z"]:
                direc_covariances[direc] = np.mean(np.mean(two_point_covariance[direc], axis=0), axis=0)[: Nr]
            s2_average = (direc_covariances['x'] + direc_covariances['y'] + direc_covariances['z'])/3
            s2_list.append(s2_average)
            f2_list.append(cal_fn(s2_average, n = 2))
        # s2--------------------
        df_s2_list = []
        for k in np.arange(0, len(s2_list)):
            df_s2_list.append(pd.DataFrame(s2_list[k], columns = ['s2']))
        df_s2 = pd.concat(df_s2_list)
        df_s2['r'] = df_s2.index
        df_s2_grouped = df_s2.groupby(['r']).agg( {'s2': [np.mean, np.std, np.size] } )
        
        # f2--------------------
        df_f2_list = []
        for k in np.arange(0, len(f2_list)):
            df_f2_list.append(pd.DataFrame(f2_list[k], columns = ['f2']))
        df_f2 = pd.concat(df_f2_list)
        df_f2['r'] = df_f2.index
        df_f2_grouped = df_f2.groupby(['r']).agg( {'f2': [np.mean, np.std, np.size] } )

        return df_s2_grouped, df_f2_grouped
    
    elif len(images.shape) == 4 and directional:

        s2_list_x = []
        s2_list_y = []
        s2_list_z = []

        for i in tqdm( range(images.shape[0]) ):

            two_point_covariance = {}
            for j, direc in enumerate(["x", "y", "z"]) :
                two_point_direc = two_point_correlation3D(images[i], j, var = 1)
                two_point_covariance[direc] = two_point_direc

            Nr = two_point_covariance[direc].shape[0]// 2
            direc_covariances = {}
            for direc in ["x", "y", "z"]:
                direc_covariances[direc] = np.mean(np.mean(two_point_covariance[direc], axis=0), axis=0)[: Nr]

            s2_list_x.append(direc_covariances['x'])
            s2_list_y.append(direc_covariances['y'])
            s2_list_z.append(direc_covariances['z'])

        # x-direction--------------------
        df_list_x = []
        for k in np.arange(0, len(s2_list_x)):
            df_list_x.append(pd.DataFrame(s2_list_x[k], columns = ['s2']))
        df_x = pd.concat(df_list_x)
        df_x['r'] = df_x.index
        df_x_grouped = df_x.groupby(['r']).agg( {'s2': [np.mean, np.std, np.size] } )

        # y-direction--------------------
        df_list_y = []
        for k in np.arange(0, len(s2_list_y)):
            df_list_y.append(pd.DataFrame(s2_list_y[k], columns = ['s2']))
        df_y = pd.concat(df_list_y)
        df_y['r'] = df_y.index
        df_y_grouped = df_y.groupby(['r']).agg( {'s2': [np.mean, np.std, np.size] } )

        # z-direction--------------------
        df_list_z = []
        for k in np.arange(0, len(s2_list_z)):
            df_list_z.append(pd.DataFrame(s2_list_z[k], columns = ['s2']))
        df_z = pd.concat(df_list_z)
        df_z['r'] = df_z.index
        df_z_grouped = df_z.groupby(['r']).agg( {'s2': [np.mean, np.std, np.size] } )

        return df_x_grouped, df_y_grouped, df_z_grouped, (df_x_grouped +  df_y_grouped + df_z_grouped)/3

def REV(image: np.ndarray,
            img_size_list: List[int],
            n_rand_samples: int)-> Dict[str, pd.DataFrame]:
    
    """
    This function receives a 3D (XCT) image and calculates average S2 and F2 for the whole image and a number of random subvolumes.
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

def RES(image: np.ndarray,
            img_size_list: List[int],
            n_rand_samples: int)-> Dict[str, pd.DataFrame]:
    
    """
    This function receives a 2D binary image and calculates average S2 and F2 for the whole image and a number of random crops.
    These average correlation functions can then be analysed to determine the RES for the image.
    Parameters
    ----------
    image: np.ndarray
    This is the 2D binary image read as numpy array to do RES analysis on.

    img_size_list: List
    list of image sizes to calculate correlation functions. These sizes should be smaller than the whole image.

    n_rand_samples: int
    number of random images used for calculating REV. use 30 or more.

    Returns
    --------
    It returns two dictionary: one for s2 (s2_dict) and one for f2 (f2_dict)
"""
    
    x_max, y_max = image.shape[:]
    
    s2_dict = {}
    f2_dict = {}


    for image_size in tqdm(img_size_list):

        all_crops = np.zeros((n_rand_samples, image_size, image_size), dtype = np.uint8)
        for i in range(n_rand_samples):

            x = np.random.randint (0, x_max - image_size)
            y = np.random.randint (0, y_max - image_size)

            crop_image = image[x:x + image_size, y:y + image_size]
            all_crops[i] = crop_image
            
        df_s2, df_f2 = calculate_two_point_df(all_crops)
        s2_dict[f'sub_{image_size}'] = df_s2
        f2_dict[f'sub_{image_size}'] = df_f2

        print(f'Image of size {image_size} done !')
        
    return s2_dict, f2_dict