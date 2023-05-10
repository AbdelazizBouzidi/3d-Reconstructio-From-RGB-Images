# Train-3D

This is a repo to complete the original work presented in
[Adelaidepth repo](https://github.com/aim-uofa/AdelaiDepth/tree/main/LeReS) with 3-d models training and data preparation
===
## What does it do
3D reconstruction from a single image is a very challenging task. State-of-the-art techniques are capable of providing accurate approximations of depth values at each pixel, which give an information about one of the required 3 dimensions.

best performing models as the one proposed in the orginal paper [the orginal paper](https://arxiv.org/pdf/2208.13241.pdf) uses huge ammount of data comming from various data, to guarentee generalization capablities and to make the training possible, the model need to be trained with normalized raw data (relative depths).

This obstacle makes 3D reconstruction through projective geometry incorrect, even if the camera parameters are provided. This is due to the fact that the model provides depth values that are invariant to scale and shift. On the other hand, 3D reconstruction through the inverse camera projection matrix can be done even if the scale is unknown, since x and y will get the same scale as z (the depth). It is possible to estimate x and y using the following equations:
'''
$$ x = {u  \over f}d $$ 
'''
'''
$$ y = {v  \over f}d $$ 
'''
'''
$$ z = d $$ 
'''
such as $u$ and $v$ are the coordinates in pixels, $f$ is the focal lenght and $d$ is the depth value. 

This linear relationship enforces the fact that $x$ and $y$ will be scaled according to the depth scale. This means that the final reconstruction, even if the scale of the depth values isn't in real meters, will manage to preserve the geometrical attributes of the scene. However, a shift in the depth values isn't tolerated, and a correctly scaled shift value is needed to recover the 3D scene. This is because the depth values are physical quantities, and altering this information will also require another set of image coordinates, other than the ones present in the RGB image at our disposal. In other words, to maintain the matching possibility between real coordinates and image coordinates, the correct shift value must be determined. We also assume that camera intrinsics aren't provided since we want the solution to be as general as possible and work with any given RGB image.

At this point, we have identified three key features needed to use relative depths for a correct 3D reconstruction: the scale, the shift, and the camera intrinsics. The scale is less important since it doesn't affect the unprojection process.

To recover the focal length and the correct shift, we aim to use the geometrical distortions to reason about the correct parameters to obtain a more realistic and geometrically-correct scene.
<table>
  <tr>
    <td>correct shift and focal length</td>
     <td>incorrect shift and focal length</td>
  </tr>
  <tr>
    <td><img src="images/correct.gif" height=300></td>
    <td><img src="images/incorrect.gif" height=300></td>
  </tr>
 </table>   

 <table>
  <tr>
    <td>correct shift and focal length</td>
     <td>incorrect shift and focal length</td>
  </tr>
  <tr>
    <td><img src="images/correct1.gif" height=300></td>
    <td><img src="images/incorrect1.gif" height=300></td>
  </tr>
 </table>   

in these examples we show how the 3D cloudpoint is affected by an incorrect shift and focal lenght, multiple distortions are preceived especially in planar objects, the models in this repo are trained to take the distorted cloudpoint as an input and output differential shift and focal lenght values to rectify the incorrect parameters used for the construction. 
## Install & Dependencies

## Dataset Preparation
| Dataset |
| ---     |
| [ScanNet](http://www.scan-net.org/) |
| [3d-ken-burns](https://github.com/sniklaus/3d-ken-burns)|
| [Taskonomy](http://taskonomy.stanford.edu/)| |

## How to use
### Launching the data preparation script
for data preparation, the script supports handling data from all the previously mentioned datasets. Assuming you are keeping the original dataset structures, you can use:
  ```bash
  python data_gen.py -ss path_to_scanNet -sk path_to_kenBurns -st path_to taskanomy -ns smaples_from_scanNet,smaples_from_kenBurns,smaples_from_taskonomy --out outputs_path
  ```
  - ns (a consecutive 3 integers) allows the user to chose the number of samples to take from each dataset, the samples will be equally sampled from all the scenes from each dataset

    - inputs structure

      ```
      |—— path_to_scanet
      |    |—— scans
      |    |     |—— sceneXX
      |    |     |     |—— sceneXX.sens
      |    |     |     |
      |—— path_to_kenBurns
      |    |—— depths
      |    |     |—— sceneXX
      |    |     |     |—— scanXX.exr
      |    |—— rgb
      |          |—— sceneXX
      |          |     |—— scanXX.json
      |          |     |
      |—— path_to_taskanomy
      |    |—— depth_zbuffer
      |    |     |—— taskonomy
      |    |     |     |—— sceneXX
      |    |     |     |      |—— scanXX.png
      |    |—— point_info
      |          |—— taskonomy
      |          |     |—— sceneXX
      |          |     |      |—— scanXX.json
      |
      ```
   - outputs structure
      ```
      |—— outputs_path
      |    |—— depths
      |    |     |—— xxxx.npz
      |    |—— intrinsics
      |    |     |—— xxxx.npy
      ```
-  Additionally at the end of the generation process, we provide two arrays in .npy format to study the distrubution of the scale and the focal lenght in your final dataset.

### Launching the training script

for the training, you will need the data generated by the previous step and to indicate your hyperparameters :
  ```bash
  python train.py --data /home/abdelaziz/workspace/trials --model_name "scale_model_uv" --input_chanels 5 --targets 1 --batch_size 20 --learning_rate 0.01 --voxel_size 0.005 --train_split 0.9 --learning_rate_decay 0.5 --epochs 5 --print_losses_evry 150 --num_w 8 --logs
  ```
  - a list of the arguemts that can be used:
    ```
    --data DATA           path to training dataset
    --model_name MODEL_NAME
    --input_chanels {3,5}
                          specify the features dimentions, 3 if your are using only x, y and z, 5 for x,y,z and u,v
    --targets {0,1,2}     a variable needed by the data loader to specify what targets to train, 0 for the shift, 1 for the scale and 2 for focal_length
    --batch_size BATCH_SIZE
    --learning_rate LEARNING_RATE
    --distort_s           indicate to the data loader whether to apply distortion to the cloudpoint through a applying a shift to the depth values, default is
                          False
    --distort_f           indicate to the data loader whether to apply distortion to the cloudpoint through an incorrect focal, default is False
    --voxel_size VOXEL_SIZE
    --init_scale INIT_SCALE
    --train_split TRAIN_SPLIT
                          percentage / 100 of data used for training, by default 1 - train_split represent the portion of data used for validation
    --learning_rate_decay LEARNING_RATE_DECAY
                          used for a linear learaning rate decay
    --epochs EPOCHS
    --logs
    --print_losses_evry PRINT_LOSSES_EVRY
    --num_w NUM_W
    ```

## Code Details
### Tested Platform
- software
  ```
  OS: Ubuntu 22.04.2 LTS
  Python: 3.10 (miniconda)
  PyTorch-Cuda: 1.13.1 - 11.7
  ```
- hardware
  ```
  CPU: Intel® Core™ i7-9700KF CPU @ 3.60GHz × 8
  GPU: Nvidia RTX 2080 SUPER 8GB
  ```
