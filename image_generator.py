import blenderproc as bproc
import argparse
import os
import numpy as np
import bpy
import random

parser = argparse.ArgumentParser()
parser.add_argument('bop_parent_path', nargs='?', help="Path to the bop datasets parent directory")
parser.add_argument('cc_textures_path', nargs='?', default="resources/cctextures", help="Path to downloaded cc textures")
parser.add_argument('output_dir', nargs='?', help="Path to where the final files will be saved ")
parser.add_argument('--num_scenes', type=int, default=2, help="How many scenes with 25 images each to generate")
args = parser.parse_args()

bproc.init()

# load a random sample of bop objects into the scene
rook = bproc.loader.load_obj("/home/max/Documents/blenderproc/datasets/models/chess/models/obj_000001.ply")[0]
queen = bproc.loader.load_obj("/home/max/Documents/blenderproc/datasets/models/chess/models/obj_000002.ply")[0]
pawn = bproc.loader.load_obj("/home/max/Documents/blenderproc/datasets/models/chess/models/obj_000003.ply")[0]
king = bproc.loader.load_obj("/home/max/Documents/blenderproc/datasets/models/chess/models/obj_000004.ply")[0]
horse = bproc.loader.load_obj("/home/max/Documents/blenderproc/datasets/models/chess/models/obj_000005.ply")[0]
bishop = bproc.loader.load_obj("/home/max/Documents/blenderproc/datasets/models/chess/models/obj_000006.ply")[0]

sampled_bop_objs = [rook,queen,pawn, king, horse, bishop]
#sampled_bop_objs = [rook,queen]

#sampled_bop_objs = bproc.loader.load_obj("/home/max/Documents/blenderproc/datasets/models/lm/models/obj_000001.ply")[0]

# Use vertex color for texturing
for i, obj in enumerate(sampled_bop_objs):
    obj.set_scale([0.001, 0.001, 0.001])
    obj.set_cp("category_id", i)
    obj.set_cp("bop_dataset_name", 'lm')

# load distractor bop objects

distractor_bop_objs = bproc.loader.load_bop_objs(bop_dataset_path = os.path.join(args.bop_parent_path, 'lm'),
                                      mm2m = True,
                                      sample_objects = True,
                                      num_of_objs_to_sample = 0)

# load BOP datset intrinsics
# bproc.loader.load_bop_intrinsics(bop_dataset_path = os.path.join(args.bop_parent_path, "lm"))
fx = 1069.86
fy = 1069.81
cx = 929.96
cy = 540.947

K = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0, 0, 1]])

#bproc.camera.set_intrinsics_from_K_matrix([[fx, 0.0, cx],[0.0,fy, cy], [0.0, 0.0, 1.0]], 640, 480)
bproc.loader.load_bop_intrinsics(bop_dataset_path = os.path.join(args.bop_parent_path, "lm"))

cc_textures = bproc.loader.load_ccmaterials(args.cc_textures_path)

# Get a list of all files in the textures directory
textures_dir = "/home/max/Documents/blenderproc/datasets/models/chess/textures"
textures_objects_list = os.listdir(textures_dir)

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

# activate depth rendering
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

for i in range(args.num_scenes):

    # set shading and physics properties and randomize PBR materials
    for obj in sampled_bop_objs:
        obj.set_shading_mode('auto')
        
    set_materials(sampled_bop_objs[0])
    set_materials(sampled_bop_objs[1])
    set_materials(sampled_bop_objs[2])
    set_materials(sampled_bop_objs[3])
    set_materials(sampled_bop_objs[4])
    set_materials(sampled_bop_objs[5])


    # sample CC Texture and assign to room planes
    random_cc_texture = np.random.choice(cc_textures)
    for plane in room_planes:
        plane.replace_materials(random_cc_texture)

    # Sample two light sources
    light_plane_material.make_emissive(emission_strength=np.random.uniform(3,6), 
                                    emission_color=np.random.uniform([0.5, 0.5, 0.5, 1.0], [1.0, 1.0, 1.0, 1.0]))  
    light_plane.replace_materials(light_plane_material)
    light_point.set_color(np.random.uniform([0.5,0.5,0.5],[1,1,1]))
    location = bproc.sampler.shell(center = [0, 0, 0], radius_min = 1, radius_max = 1.5,
                            elevation_min = 5, elevation_max = 89)
    light_point.set_location(location)
    
    # Define a function that samples the initial pose of a given object above the ground
    def sample_initial_pose(obj: bproc.types.MeshObject):
        obj.set_location(bproc.sampler.upper_region(objects_to_sample_on=room_planes[0:1],
                                                    min_height=1, max_height=4, face_sample_range=[0.4, 0.6]))
        obj.set_rotation_euler(np.random.uniform([0, 0, 0], [0, 0, np.pi * 2]))

    # Sample objects on the given surface
    placed_objects = bproc.object.sample_poses_on_surface(objects_to_sample=sampled_bop_objs,
                                            surface=room_planes[0],
                                            sample_pose_func=sample_initial_pose,
                                            min_distance=0.01,
                                            max_distance=0.2)

    # BVH tree used for camera obstacle checks
    bop_bvh_tree = bproc.object.create_bvh_tree_multi_objects(placed_objects)

    poses = 0
    while poses < 10:
        # Sample location
        location = bproc.sampler.shell(center = [0, 0, 0],
                                radius_min = 0.35,
                                radius_max = 1.5,
                                elevation_min = 5,
                                elevation_max = 89,
                                uniform_volume = False)

        # Determine point of interest in scene as the object closest to the mean of a subset of objects
        poi = bproc.object.compute_poi(np.random.choice(placed_objects,size=len(placed_objects), replace=False))
        # Compute rotation based on vector going from location towards poi
        rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location, inplane_rot=np.random.uniform(-0.7854, 0.7854))
        # Add homog cam pose based on location an rotation
        cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
        proximity_checks = {"min": 0.2}

        # Check that obstacles are at least 0.3 meter away from the camera and make sure the view interesting enough
        if bproc.camera.perform_obstacle_in_view_check(cam2world_matrix, proximity_checks, bop_bvh_tree) and bproc.camera.scene_coverage_score(cam2world_matrix) > 0.3:
            # Persist camera pose
            bproc.camera.add_camera_pose(cam2world_matrix)
            poses += 1



    # render the whole pipeline
    data = bproc.renderer.render()

    # Write data in bop format
    bproc.writer.write_bop(args.output_dir, sampled_bop_objs, data["depth"], data["colors"], m2mm=True, append_to_existing_output=True)
    