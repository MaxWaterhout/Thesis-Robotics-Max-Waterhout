# This is the thesis github repo of Max Waterhout
This is the GitHub repo for the thesis of Max Waterhout for my master Robotics at the Technical University of Delft. The full thesis can be read at: https://repository.tudelft.nl/islandora/object/uuid%3A4506c969-220d-4257-ad5b-91853ea83ccb?collection=education
# Creating synthetic data for 6D pose estimation
## Prerequisites
Before running the code, make sure you have the following prerequisites installed:

1. Install BlenderProc using the command: `pip install blenderproc` or clone it directly from the GitHub repository: `https://github.com/DLR-RM/BlenderProc`.
2. Test if BlenderProc is correctly installed with `blenderproc quickstart`. It also downloads Blender for you. 
3. Download textures by running the following command in the command line: `blenderproc download cc_textures`. Place the textures in de `datasets` folder.
4. Download the BOP toolkit by cloning the repository: `git clone https://github.com/thodan/bop_toolkit.git`. Place the toolkit in your working folder.

Ensure that you have completed these prerequisites before proceeding with the code.

## Creation of 3D CAD models
To generate synthetic data, we need CAD models to be inserted into the scene. If you don't have the CAD model, you can follow these steps:

1. Use a scanner (In this work the ARTEC EVA-S 3D Scanner is used) to create a CAD model.
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

## Training the 6D Pose estimation model

1. Read the paper "EfficientPose: An efficient, accurate and scalable end-to-end 6D multi object pose estimation approach" and clone its repo at:  https://github.com/ybkscht/EfficientPose
2. Put the images in LineMod format. This way EfficientPose can be trained on them. 
3. Follow the instructions on the EfficientPose GitHub page to train the models. 

## Picking with a Franka Robot
1. First get the robot to work: https://frankaemika.github.io/docs/overview.html
2. `ROS/picking_ws` contains the ROS network for the picking pipeline. The whole pipeline is explained in the thesis itself. An example command for picking is: `roslaunch franka_scripts pick_and_place.launch robot_ip:=192.168.0.200 real_robot:=true robot:=fr3 use_rviz:=False`
