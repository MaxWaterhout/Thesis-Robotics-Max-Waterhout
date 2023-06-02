import blenderproc as bproc
import argparse
import os
import numpy as np
import time 
import bpy

start = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('bop_parent_path', help="Path to the bop datasets parent directory")
parser.add_argument('cc_textures_path', default="resources/cctextures", help="Path to downloaded cc textures")
parser.add_argument('output_dir', help="Path to where the final files will be saved ")
parser.add_argument('--num_scenes', type=int, default=2, help="How many scenes with 25 images each to generate")
#parser.add_argument('--obj_id', type=int, default=1, help="Images of which object?")
args = parser.parse_args()

bproc.init()

# load bop objects into the scene
#target_bop_objs = bproc.loader.load_bop_objs(bop_dataset_path = os.path.join(args.bop_parent_path, 'lm'), mm2m = True,obj_ids=[args.obj_id])
target_bop_objs = bproc.loader.load_obj("/home/max/Documents/blenderproc/datasets/models/lm/models/obj_000001.ply")[0]

# Use vertex color for texturing

target_bop_objs.set_scale([0.001, 0.001, 0.001])

"""dist_bop_objs_horse = bproc.loader.load_blend("/home/max/Documents/blenderproc/chess_pieces/blender/horse.blend")
dist_bop_objs_pawn = bproc.loader.load_blend("/home/max/Documents/blenderproc/chess_pieces/blender/pawn.blend")
dist_bop_objs_queen = bproc.loader.load_blend("/home/max/Documents/blenderproc/chess_pieces/blender/queen.blend")
dist_bop_objs_king = bproc.loader.load_blend("/home/max/Documents/blenderproc/chess_pieces/blender/king.blend")
dist_bop_objs_bishop = bproc.loader.load_blend("/home/max/Documents/blenderproc/chess_pieces/blender/bishop.blend")

dist_bop_objs = [dist_bop_objs_horse[0] ,dist_bop_objs_pawn[0] ,dist_bop_objs_queen[0] ,dist_bop_objs_king[0] ,dist_bop_objs_bishop[0]]"""

#ycbv_dist_bop_objs = bproc.loader.load_bop_objs(bop_dataset_path = os.path.join("datasets/models", 'ycbv'), mm2m = True)
# load BOP dataset intrinsics
#bproc.loader.load_bop_intrinsics(bop_dataset_path = os.path.join(args.bop_parent_path, 'lm'))
fx = 1069.86
fy = 1069.81
cx = 929.96
cy = 540.947

K = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0, 0, 1]])

bproc.loader.load_bop_intrinsics(bop_dataset_path = os.path.join(args.bop_parent_path, "lm"))


# set shading and hide objects  
#for obj in (target_bop_objs):
target_bop_objs.set_shading_mode('auto')
target_bop_objs.hide(True)
"""    
for obj in (dist_bop_objs):
    obj.set_shading_mode('auto')
    obj.hide(True)"""

#create room
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

# activate depth rendering without antialiasing and set amount of samples for color rendering
bproc.renderer.enable_depth_output(activate_antialiasing=False)
bproc.renderer.set_max_amount_of_samples(50)

target_bop_objs.set_cp("category_id", 1)
print(target_bop_objs)
start_2 = time.time()

for i in range(args.num_scenes):

    # Sample bop objects for a scene
    
    #print(sampled_target_bop_objs,'target')
    
    # Randomize materials and set physics
    #for obj in (sampled_target_bop_objs):
    image = bpy.data.images.load(filepath="/home/max/Documents/blenderproc/datasets/models/lm/models/obj_000001.jpg")

    mat = target_bop_objs.get_materials()[0]
    # Set it as base color of the current material
    mat.set_principled_shader_value("Base Color", image)
    mat.set_principled_shader_value("Roughness", np.random.uniform(0, 1.0))
    mat.set_principled_shader_value("Specular", np.random.uniform(0, 1.0))
    target_bop_objs.hide(False)
    
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
              
    # Define a function that samples the initial pose of a given object above the ground
    

    def sample_initial_pose(obj: bproc.types.MeshObject):
        obj.set_location(bproc.sampler.upper_region(objects_to_sample_on=room_planes[0:1],
                                                    min_height=1, max_height=4, face_sample_range=[0.4, 0.6]))
        obj.set_rotation_euler(np.random.uniform([0, 0, 0], [0, 0, np.pi * 2]))
    
    # Sample objects on the given surface
    placed_objects = bproc.object.sample_poses_on_surface(objects_to_sample=[target_bop_objs],
                                                          surface=room_planes[0],
                                                          sample_pose_func=sample_initial_pose,
                                                          min_distance=0.01,
                                                          max_distance=1)
    
    
    # BVH tree used for camera obstacle checks
    bop_bvh_tree = bproc.object.create_bvh_tree_multi_objects(placed_objects)

    cam_poses = 0

    # Enable physics for spheres (active) and the surface (passive)
    for obj in placed_objects:
        obj.enable_rigidbody(True)
    for plane in room_planes:
        plane.enable_rigidbody(False)

    # Run the physics simulation
    
    bproc.object.simulate_physics_and_fix_final_poses(min_simulation_time=2, max_simulation_time=4, check_object_interval=1)
    while cam_poses < 1:
        
        # Sample location
        

        location = np.random.uniform([-10, -10, 8], [10, 10, 12])

        # Determine point of interest in scene as the object closest to the mean of a subset of objects
        poi = bproc.object.compute_poi(placed_objects)

        #poi = bproc.object.compute_poi(np.random.choice(placed_objects,size=len(placed_objects), replace=False))
        # Compute rotation based on vector going from location towards poi
        rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location, inplane_rot=np.random.uniform(-0.7854, 0.7854))
        # Add homog cam pose based on location an rotation
        cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
        
        # Check that obstacles are at least 0.3 meter away from the camera and make sure the view interesting enough
        if bproc.camera.perform_obstacle_in_view_check(cam2world_matrix, {"min": 0.005}, bop_bvh_tree):
            # Persist camera pose
            bproc.camera.add_camera_pose(cam2world_matrix, frame=cam_poses)
            cam_poses += 1
    
    
    data = bproc.renderer.render()
    # Write data in bop format
    bproc.writer.write_bop(os.path.join(args.output_dir, 'bop_data'),
                           depth_scale = 0.1,
                           depths = data["depth"],
                           colors = data["colors"], 
                           color_file_format = "PNG",
                           ignore_dist_thres = 10)

    
    for obj in (sampled_target_bop_objs):      
        obj.hide(True)
    
end = time.time()
print(round(end-start_2)," seconds elapsed for images")
print(round(end-start)," seconds elapsed")
