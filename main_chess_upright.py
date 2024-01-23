import blenderproc as bproc
import argparse
import os
import numpy as np
import random
import bpy
import time 

start = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('bop_parent_path', help="Path to the bop datasets parent directory")
parser.add_argument('cc_textures_path', default="datasets/cc_textures", help="Path to downloaded cc textures")
parser.add_argument('output_dir', help="Path to where the final files will be saved ")
parser.add_argument('--num_scenes', type=int, default=2000, help="How many scenes with 25 images each to generate")
args = parser.parse_args()

bproc.init()

current_dir = os.getcwd()  # Get the current working directory

# Concatenate the current directory with the relative paths to the models
rook_path = os.path.join(current_dir, "datasets/models/chess/models/obj_000001.ply")
queen_path = os.path.join(current_dir, "datasets/models/chess/models/obj_000002.ply")
pawn_path = os.path.join(current_dir, "datasets/models/chess/models/obj_000003.ply")
king_path = os.path.join(current_dir, "datasets/models/chess/models/obj_000004.ply")
horse_path = os.path.join(current_dir, "datasets/models/chess/models/obj_000005.ply")
bishop_path = os.path.join(current_dir, "datasets/models/chess/models/obj_000006.ply")

# Load the objects using the updated paths
rook = bproc.loader.load_obj(rook_path)[0]
queen = bproc.loader.load_obj(queen_path)[0]
pawn = bproc.loader.load_obj(pawn_path)[0]
king = bproc.loader.load_obj(king_path)[0]
horse = bproc.loader.load_obj(horse_path)[0]
bishop = bproc.loader.load_obj(bishop_path)[0]

target_bop_objs = [rook,queen,pawn, king, horse, bishop]

for i, obj in enumerate(target_bop_objs):
    obj.set_scale([0.001, 0.001, 0.001])
    obj.set_cp("category_id", i+1)
    obj.set_cp("bop_dataset_name", 'lm')



# load distractor bop objects
#tless_dist_bop_objs = bproc.loader.load_bop_objs(bop_dataset_path = os.path.join(args.bop_parent_path, 'tless'), model_type = 'cad', mm2m = True)
ycbv_dist_bop_objs = bproc.loader.load_bop_objs(bop_dataset_path = os.path.join(args.bop_parent_path, 'ycbv'), mm2m = True)
lm_dist_bop_objs = bproc.loader.load_bop_objs(bop_dataset_path = os.path.join(args.bop_parent_path, 'lm'), mm2m = True)

# load BOP datset intrinsics
bproc.loader.load_bop_intrinsics(bop_dataset_path = os.path.join(args.bop_parent_path, 'lm'))

textures_dir = os.path.join(current_dir, "datasets/models/chess/textures")

#textures_dir = "/home/max/Documents/blenderproc/datasets/models/chess/textures"
textures_objects_list = os.listdir(textures_dir)

# set shading and hide objects
for obj in (target_bop_objs + ycbv_dist_bop_objs +lm_dist_bop_objs ):
    obj.set_shading_mode('auto')
    obj.hide(True)
    
# create room
room_planes = [bproc.object.create_primitive('PLANE', scale=[2, 2, 1]),
               bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[0, -2, 2], rotation=[-1.570796, 0, 0]),
               bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[0, 2, 2], rotation=[1.570796, 0, 0]),
               bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[2, 0, 2], rotation=[0, -1.570796, 0]),
               bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[-2, 0, 2], rotation=[0, 1.570796, 0])]

# sample light color and strenght from ceiling
light_plane = bproc.object.create_primitive('PLANE', scale=[3, 3, 1], location=[0, 0, 10])
light_plane.set_name('light_plane')
light_plane_material = bproc.material.create('light_material')

# sample point light on shell
light_point = bproc.types.Light()
light_point.set_energy(200)

# load cc_textures
cc_textures = bproc.loader.load_ccmaterials(args.cc_textures_path)

# Define a function that samples 6-DoF poses
def sample_pose_func(obj: bproc.types.MeshObject):
    min = np.random.uniform([-0.3, -0.3, 0.0], [-0.2, -0.2, 0.0])
    max = np.random.uniform([0.2, 0.2, 0.4], [0.3, 0.3, 0.6])
    obj.set_location(np.random.uniform(min, max))
    obj.set_rotation_euler(bproc.sampler.uniformSO3())
    
# activate depth rendering without antialiasing and set amount of samples for color rendering
bproc.renderer.enable_depth_output(activate_antialiasing=False)
bproc.renderer.set_max_amount_of_samples(50)

def set_materials(obj):
    obj.set_shading_mode('auto')
    mat = obj.get_materials()[0]
    # Choose a random file from the list
    random_file = random.choice(textures_objects_list)
    # Construct the full path to the randomly chosen file
    random_texture_object = os.path.join(textures_dir, random_file)

    image = bpy.data.images.load(filepath=random_texture_object)
    # Set it as base color of the current material
    mat.set_principled_shader_value("Base Color", image)      
    mat.set_principled_shader_value("Roughness", np.random.uniform(0, 1.0))
    mat.set_principled_shader_value("Specular", np.random.uniform(0, 1.0))
    obj.hide(False)

end = time.time()
for i in range(args.num_scenes):
    print(i,' Of scene ', args.num_scenes)
    # Sample bop objects for a scene
    #sampled_target_bop_objs = list(np.random.choice(target_bop_objs, size=3))
    #sampled_target_bop_objs = list(np.random.choice(target_bop_objs, size=15, replace=True))

    sampled_target_bop_objs = target_bop_objs
    
    sampled_distractor_bop_objs = list(np.random.choice(ycbv_dist_bop_objs, size=3, replace=False))
    sampled_distractor_bop_objs += list(np.random.choice(lm_dist_bop_objs, size=3, replace=False))

    # Randomize materials and set physics
    for obj in (sampled_target_bop_objs + sampled_distractor_bop_objs):        
        mat = obj.get_materials()[0]      
        mat.set_principled_shader_value("Roughness", np.random.uniform(0, 1.0))
        mat.set_principled_shader_value("Specular", np.random.uniform(0, 1.0))
        obj.hide(False)

    set_materials(sampled_target_bop_objs[0])
    set_materials(sampled_target_bop_objs[1])
    set_materials(sampled_target_bop_objs[2])
    set_materials(sampled_target_bop_objs[3])
    set_materials(sampled_target_bop_objs[4])
    set_materials(sampled_target_bop_objs[5])
    
    # Sample two light sources
    light_plane_material.make_emissive(emission_strength=np.random.uniform(3,6), 
                                    emission_color=np.random.uniform([0.5, 0.5, 0.5, 1.0], [1.0, 1.0, 1.0, 1.0]))  
    light_plane.replace_materials(light_plane_material)
    light_point.set_color(np.random.uniform([0.5,0.5,0.5],[1,1,1]))
    location = bproc.sampler.shell(center = [0, 0, 0], radius_min = 1, radius_max = 1.5,
                            elevation_min = 5, elevation_max = 89)
    light_point.set_location(location)

    # sample CC Texture and assign to room planes
    random_cc_texture = np.random.choice(cc_textures)
    for plane in room_planes:
        plane.replace_materials(random_cc_texture)


    # Sample object poses and check collisions 
    bproc.object.sample_poses(objects_to_sample = sampled_target_bop_objs + sampled_distractor_bop_objs,
                            sample_pose_func = sample_pose_func, 
                            max_tries = 1000)
            
    # Define a function that samples the initial pose of a given object above the ground
    def sample_initial_pose(obj: bproc.types.MeshObject):
        obj.set_location(bproc.sampler.upper_region(objects_to_sample_on=room_planes[0:1],
                                                    min_height=1, max_height=4, face_sample_range=[0.4, 0.6]))
        obj.set_rotation_euler(np.random.uniform([0, 0, 0], [0, 0, np.pi * 2]))

    # Sample objects on the given surface
    placed_objects = bproc.object.sample_poses_on_surface(objects_to_sample=sampled_target_bop_objs + sampled_distractor_bop_objs,
                                                          surface=room_planes[0],
                                                          sample_pose_func=sample_initial_pose,
                                                          min_distance=0.01,
                                                          max_distance=0.4)

    # BVH tree used for camera obstacle checks
    bop_bvh_tree = bproc.object.create_bvh_tree_multi_objects(sampled_target_bop_objs + sampled_distractor_bop_objs)

    cam_poses = 0
   

    while cam_poses < 10:
        # Sample location
        location = bproc.sampler.shell(center = [0, 0, 0],
                                radius_min = 0.2,
                                radius_max = 0.5,
                                elevation_min = 5,
                                elevation_max = 89)
        # Determine point of interest in scene as the object closest to the mean of a subset of objects
        poi = bproc.object.compute_poi(np.random.choice(sampled_target_bop_objs, size=6, replace=False))
        # Compute rotation based on vector going from location towards poi
        rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location, inplane_rot=np.random.uniform(-0.7854, 0.7854))
        # Add homog cam pose based on location an rotation
        cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
        
        # Check that obstacles are at least 0.3 meter away from the camera and make sure the view interesting enough
        if bproc.camera.perform_obstacle_in_view_check(cam2world_matrix, {"min": 0.1}, bop_bvh_tree):
            # Persist camera pose
            bproc.camera.add_camera_pose(cam2world_matrix, frame=cam_poses)
            cam_poses += 1

    # render the whole pipeline
    data = bproc.renderer.render()

    # Write data in bop format
    bproc.writer.write_bop(os.path.join(args.output_dir, 'bop_data'),
                           target_objects = sampled_target_bop_objs,
                           dataset = 'lm',
                           depth_scale = 0.1,
                           depths = data["depth"],
                           colors = data["colors"], 
                           color_file_format = "JPEG",
                           ignore_dist_thres = 10,
                           frames_per_chunk=69999
                           )
    
    for obj in (sampled_target_bop_objs + sampled_distractor_bop_objs):      
        obj.hide(True)

end_end = time.time()
print(round(end-start),'loading seconds')
print(round(end_end-end),'scenes rendering')
print(round(end_end-start),'full time')

    
