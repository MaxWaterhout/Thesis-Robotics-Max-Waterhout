# Creating synthetic data for 6D pose estimation
## Prerequisites
Before running the code, make sure you have the following prerequisites installed:

1. Install BlenderProc using the command: `pip install blenderproc` or clone it directly from the GitHub repository: `https://github.com/DLR-RM/BlenderProc`.
2. Download the BOP toolkit by cloning the repository: `git clone https://github.com/thodan/bop_toolkit.git`. Place the toolkit in your working folder.
3. Download Blender from the official website: `https://www.blender.org/download/`.
4. Download textures by running the following command in the command line: `blenderproc download cc_textures`. Place the textures in de `datasets` folder.

Ensure that you have completed these prerequisites before proceeding with the code.

## Creation of 3D CAD models
To generate synthetic data, we need CAD models to be inserted into the scene. If you don't have the CAD model, you can follow these steps:

1. Use the ARTEC EVA-S 3D Scanner to create a model.
2. Load the model in Blender. Ensure that the model is in meters since the script expects it to be in that unit.
3. Align the object frame with the world frame. In our dataset, the Z-axis represents the sky. You can achieve this by entering edit mode and using the mesh transformation tools to align the axis. 
4. *(Optional)* Add a modifier in the Modifier properties. Choose the *decimate* option to reduce the point cloud's size and potentially improve computational costs. You can adjust the decimation ratio to your preference but we use 0.5 for a balance between point cloud detail and computational costs.
5. Convert the model into a point cloud by exporting it as Stanford (.ply) format. Make sure to select the ASCII format. Save the object in the folder `datasets/models/chess/models` with the name `obj_00000*.ply`.
6. Run the command `python bop_toolkit/scripts/calc_model_info.py` to generate the `models_info.json` file.
7. If your 3D scanner does not capture colors, such as the ARTEC EVA-S, you will need to manually select a texture for the object. In Blender, navigate to the shading tab and ensure that the model is in object-mode with viewport shading enabled. From the available textures in the datasets/cc_textures folder, choose a folder and select the *_Color.jpg file in Blender. Connect the color output to the base_color input of the Principled BSDF shader. You can now preview your object with the applied texture. In our case, we used multiple wood textures, which were placed in the datasets/models/chess/textures folder.

This process will help you create the necessary CAD models for your synthetic data.

## Creation of images
To generate images, you can execute the following command in your terminal:

 `blenderproc run main_chess_upright.py datasets/models datasets/cc_textures output --num_scenes=2`

Let's break down the arguments:

- `datasets/models`: Path to the parent directory containing the BOP datasets.
- `datasets/cc_textures`: Path to the directory containing the textures.
- `output`: Folder where the generated images will be saved.
- `--num_scenes=2`: Specifies the number of scenes to render, each with a different textured wall.

Executing this command initiates the image generation process. It will render the specified number of scenes, applying the desired textures, and save the resulting images in the output folder.



## 2: Change train_pbr to train and change lm to chess

## 2: Create masks and calculate model info with BOP toolkit (can change parameters of the config files) : python scripts/calc_gt_masks.py & python scripts/calc_gt_info.py 

python bop_toolkit/scripts/calc_gt_masks.py 
python bop_toolkit/scripts/calc_gt_info.py 

## : for different objects create symalinks ln -s /home/max/Documents/blenderproc/output/bop_data/lm/train_pbr/000000/rgb /home/max/Documents/blenderproc/output/bop_data/lm/train_pbr/000002



3: Create models info from the .ply models with: python bop_toolkit/scripts/calc_model_info.py output/bop_data . change datasets_params.py in scripts/bop_toolkit_lib and add models 

for debug: python debug.py --phi 0 --annotations linemod /home/max/Documents/blenderproc/output/bop_data/lm/train/data/ --object-id 1

blenderproc at: /home/max/anaconda3/lib/python3.9/site-packages/blenderproc/python/utility
Intrinsics left camera = 
fx = 1069.86
fy = 1069.81
cx = 929.96
cy = 540.947
k1 = -0.0458
k2 = 0.0162
p1 = -0.0001
p2 = -0.0009
k3 = -0.0068

python debug.py --annotations --draw_2d-bboxes linemod /home/max/Documents/Linemod_preprocessed/ --object-id 8
python debug.py --annotations --draw_2d-bboxes linemod /home/max/Documents/GitHub/thesis/output/bop_data/chess/ --object-id 1
python train.py --phi 0 --lr 0.001 --epochs 3 --steps 3 --gpu 0 --weights /home/max/Documents/EfficientPose/weights/object_1/phi_0_linemod_best_ADD.h5 linemod /home/max/Documents/Linemod_preprocessed --object-id 1  
python train.py --phi 0  --lr 0.0001 --epochs 100 --steps 100 --no-snapshots --gpu 0 --weights /home/max/Documents/EfficientPose/weights/efficientdet-d0.h5 linemod /home/max/Documents/GitHub/thesis/output/bop_data/chess/ --object-id 1 --no-6dof-augmentation --tensorboard-dir "home/max/Documents/EfficientPose/logs/"
python debug.py --annotations --draw_2d-bboxes linemod /home/max/Documents/GitHub/thesis/output/bop_data/chess/ --object-id 1 
use cuda10.0