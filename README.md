## True 2D-to-3D Reconstruction of Heterogenous Porous Media via Deep Generative Adversarial Networks (GANs)
 This repository provides codes for 3D image reconstructions using only 2D images scanned from orthogonal planes of a berea sandstone. The model, named SliceGAN, was originaly intoduced by [Kench and Cooper(2021)](https://github.com/stke9/SliceGAN). However, the original model was designed to use slices of a ground-truth 3D volume for training. Here we introduce several modifications to the original approach. Our contributions are:
 1) Modifying the sampling workflow and dataloaders to take random images of representative sizes from three 2D images obtained from different orientations.
 2) Adding a model evaluation to the training loop based on the calculation of two-point correlation function in 2D (for training images) and 3D for generated volumes in different planes.
 3) Changing the architecture of Generator and Discriminator to reduce the chekerboard patterns, as well as operating on larger images of resolution 256.

 ## Training workflow
 After the preparation of thin sections, we acquired large-area BSE and optical images, which are then segmented into pore and solid phases. Subsequently, we extract random crops with representative elementary size (RES) from these 2D images, taken from different orientations. To calculate the representative size from your 2D images run `RES.py` first. If this size is too large and your computational resource is not enough, you can resize it upon loading the data (see `train.py`). In our case, RES size for BSE images were 384 which were then resized on-the-fly to 256 pixels due to our GPU memory constraints. In the next step, these images are fed to the 3 discriminators for each orientation x, y, z. The discriminator also receives slices from different orientations of the generated image by the generator.
 ![](Fig1_Workflow.jpg)

 ## Usage
 Before training your model for 2D-to-3D reconstructions,representative image size should be determined. If you have a 3D ground-truth volume of your sample, run:
```
python REV.py --image_dir 'full path to your 3D binary image' \
--imag_sizes 400, 350, 256, 128, 64 \
--n_rnd_samples 50 --output_dir 'path to save the output'

# this will calculate average $S_2$ and $F_2$ in three dimensions and save them as a dictionary `*.pkl` file in the 
```

 For training the model using your 2D images. you need three (for anisotropic microstructure) 2D binary images (with value 1 showing your phase of interest). To train your own model run:
 ```
 python train.py --dir_img_1 Data\BSE_images\X_R3_binary.tif \
 --dir_img_2 Data\BSE_images\Y_R2_binary.tif \
 --dir_img_3 Data\BSE_images\Z_R1_binary.tif \
 --RES 384 --train_img_size 256 --batch_size 2

 ```

 After training, your best model will be saved in a folder in the specified directory. Then for generating images using pre-trained model:
 ```
 ## Here is an example to generate using the generator trained with BSE images:
 python inference.py --G_pkl Pretrained_models\WGAN_Gen_iter_34000.pt \
 --ngpu 1 --num_img 100 img_size 256 --z_size 4.

 ``` 
 `img_size` is the image size you used for training your model. for generating images larger than your training images you can run the inference with `z_size` of 6 to generate images of size $512^3$.

Here is the summary of scripts in this repository:
- ``REV.py``: Calculates representative elementary volume from a 3D image, if applicable.
<!-- - ``RES.py``: Calculate representative elementary size for 2D images. -->
- ``train.py``: Trains the modified SliceGAN using 2D images.
- ``inference.py``: Generate 3D images using the pre-trained generator.


 

 