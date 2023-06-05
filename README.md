# Blender creation of CAD models
The ply model needs to be in meter but CAD model needs to be in Meter so scale 1/1000, add modifier decimate and ratio to 50%. 

## 1: create dataset with: 
 ```blenderproc run image_generator.py datasets/models datasets/cc_textures output --num_scenes=1```



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