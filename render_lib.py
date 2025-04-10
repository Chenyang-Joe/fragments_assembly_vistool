import bpy
import os
import random
import json
import subprocess
from mathutils import Quaternion, Vector, Matrix
import numpy as np
import shutil
import glob
import math
import time
import trimesh


def select_model(folder_path, file_num, single_mode, random_draw, traversal_all, range, data_mode):
    if single_mode:
        return [folder_path]
    if not os.path.isdir(folder_path):
        raise ValueError(f"{folder_path} not a avaliable folder path")

    if data_mode != "puzzlefusion":
        all_files = [os.path.join(folder_path, name) for name in os.listdir(folder_path)
                    if (os.path.isfile(os.path.join(folder_path, name)) and (name[0]!="."))]
        if os.path.basename(all_files[0]).split('.')[0].isdigit():
            all_files = sorted(all_files, key=lambda x: int(os.path.basename(x).split('.')[0]))
    else:
        all_files = [os.path.join(folder_path, name) for name in os.listdir(folder_path)
            if os.path.isdir(os.path.join(folder_path, name))]
        if os.path.basename(all_files[0]).split('/')[0].isdigit():
            all_files = sorted(all_files, key=lambda x: int(os.path.basename(x).split('/')[-1]))

        
    start = range[0]
    end = range[1]+1
    if traversal_all:
        file_list = all_files
    else:
        if file_num > len(all_files):
            raise ValueError("folder_num excesses the total folder number")
        if (end-start<file_num):
            print("The file number excesses range.")
        else:
            file_list = all_files[start: start+file_num]
            print("Start at:", start, " End at:", end)

    if random_draw:
        random.shuffle(file_list)

    return file_list

def random_color():
    global colors
    if not colors:
        fill_colors()
        # raise ValueError("No colors left in the list!")
    # selected_color = random.choice(colors)  # 随机选择一个颜色
    selected_color = colors[0]
    colors.remove(selected_color)          # 从列表中删除该颜色
    return selected_color

def check_empty(trans_file_path, data_mode):
    if data_mode != "puzzlefusion":
        with open(trans_file_path, "r", encoding="utf-8") as f:
            content = f.read()
            if not content.strip():  # If the file is an empty .json, ignore it
                return True
    return False

def read_trans(trans_file_path, data_mode, gt_mode = False):
    redundant_path = None
    removal_name = None
    order_str = None
    scale_factor = 1
    if data_mode != "puzzlefusion":
        with open(trans_file_path, 'r') as f:
            data = json.load(f)
        gt_trans_rots = np.array(data['gt_trans_rots'])
        pred_trans_rots = np.array(data['pred_trans_rots'])
        # init_pose = np.array(data['init_pose'])
        init_pose = np.array([0,0,0,0,0,0,0])
        model_path_half = data['name']
        if 'redundant_pieces' in data.keys():
            redundant_path = data['redundant_pieces']
        if 'removal_pieces' in data.keys():
            removal_name = data['removal_pieces']
        if 'pieces' in data.keys():
            order_str = data['pieces']
        if 'mesh_scale' in data.keys():
            scale_factor = 1/data['mesh_scale']
    else:
        predict_pattern = "predict_*.npy"
        # Find the first file that matches the predict pattern
        predict_files = glob.glob(os.path.join(trans_file_path, predict_pattern))
        predict_file_path = predict_files[0]
        # Load the predict.npy file
        pred_trans_rots = np.load(predict_file_path)
        gt_trans_rots = np.load(f"{trans_file_path}/gt.npy")
        init_pose = np.load(f"{trans_file_path}/init_pose.npy")
        with open(f"{trans_file_path}/mesh_file_path.txt", 'r') as file:
            model_path_half = file.read().strip()

    if gt_mode:
        pred_trans_rots = pred_trans_rots[:1, :, :]*0
        gt_trans_rots = gt_trans_rots*0
        init_pose = init_pose*0

    # pred_trans_rots = np.concatenate((pred_trans_rots[:1, :, :] * 0, pred_trans_rots), axis=0)
    # pred_trans_rots = np.concatenate((np.expand_dims(gt_trans_rots, axis=0), pred_trans_rots), axis=0)

    return gt_trans_rots, pred_trans_rots, init_pose, model_path_half, redundant_path, removal_name, order_str, scale_factor

def reset_scene():
    """Reset the Blender scene by deleting all objects and clearing unused data."""
    # gc.collect()

    # Delete all objects in the scene
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    
    # Clear unused data blocks
    bpy.ops.outliner.orphans_purge(do_recursive=True)

    # bpy.ops.ed.undo_reset()
    # bpy.ops.ptcache.free_all()  
    # bpy.ops.render.clear()      
    # bpy.ops.wm.read_factory_settings(use_empty=True)





def delete_default_objects():
    """Delete default objects like Light, Camera, and Cube."""
    bpy.ops.object.select_all(action='DESELECT')
    for obj in bpy.data.objects:
        if obj.name in ['Light', 'Camera', 'Cube']:
            obj.select_set(True)
    bpy.ops.object.delete()

def parse_obj(filepath, force_painting):
    """Parse an OBJ file and extract vertices and faces."""
    vertices = []
    faces = []
    vertex_colors = []

    with open(filepath, 'r') as file:
        for line in file:
            if line.startswith('v '):
                parts = line.split()
                vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
                if len(parts) >=6 and (not force_painting):
                    r = float(parts[4])
                    g = float(parts[5])
                    b = float(parts[6])
                    vertex_colors.append((r, g, b))
                else:
                    vertex_colors = None
            elif line.startswith('f '):
                parts = line.split()
                face = [int(p.split('/')[0]) - 1 for p in parts[1:]]
                faces.append(face)

    if vertex_colors:
        vertex_0 = vertex_colors[0]
        union_color = all(np.all(vertex == vertex_0) for vertex in vertex_colors)
        if union_color:
            vertex_colors = None

    return vertices, faces, vertex_colors

def parse_ply(file_path, force_painting):
    mesh_data = trimesh.load_mesh(file_path)
    vertices = mesh_data.vertices.tolist()
    faces = mesh_data.faces.tolist()
    try:
        vc = mesh_data.visual.vertex_colors
        if vc is not None and len(vc) == len(vertices) and (not force_painting):
            color_0 = vc[0]
            union_color = all(np.all(color == color_0) for color in vc)
            if union_color:
                vertex_colors = None # all vertices use the same color, which provides no texture info
            else:
                vertex_colors = [tuple(c/255 for c in color) for color in vc]
        else:
            vertex_colors = None # not all vertices has color
    except Exception as e:
        vertex_colors = None
        print(e)
    return vertices, faces, vertex_colors



def create_final_object_list(obj_list, force_painting, folder_path = ""):
    imported_objects = []
    VFC_list = []
    file_path_list = []
    for obj_file in obj_list:
        if os.path.exists(os.path.join(folder_path, obj_file)):
            file_path = os.path.join(folder_path, obj_file)
        elif os.path.exists(os.path.join(folder_path, obj_file+'.obj')):
            file_path = os.path.join(folder_path, obj_file+'.obj')
        elif os.path.exists(os.path.join(folder_path, obj_file+'.ply')):
            file_path = os.path.join(folder_path, obj_file+'.ply')
        else:
            print("cannot recognise path", os.path.join(folder_path, obj_file))
        file_path_list.append(file_path)

        if file_path.lower().endswith(".obj"):  # check whether it is obj or ply
            vertices, faces, vertex_colors = parse_obj(file_path, force_painting)
        else:
            vertices, faces, vertex_colors = parse_ply(file_path, force_painting)
        VFC_list.append([vertices, faces, vertex_colors])

    for [vertices, faces, vertex_colors], obj in zip(VFC_list, obj_list):
        # Create a new mesh and object in Blender
        mesh = bpy.data.meshes.new(name=f"Mesh_{obj_file.split('/')[-1]}")
        mesh.from_pydata(vertices, [], faces)
        mesh.update()

        if vertex_colors is not None and len(vertex_colors) == len(vertices):
            color_layer = mesh.color_attributes.new(name="Col", domain="CORNER", type="FLOAT_COLOR")
            for poly in mesh.polygons:
                for loop_index in poly.loop_indices:
                    vertex_index = mesh.loops[loop_index].vertex_index
                    color = vertex_colors[vertex_index]
                    if len(color) == 3:
                        color = (color[0], color[1], color[2], 1.0)
                    color_layer.data[loop_index].color = color

        obj = bpy.data.objects.new(name=f"Object_{obj_file.split('/')[-1]}", object_data=mesh)
        bpy.context.collection.objects.link(obj)
        imported_objects.append(obj)
    return imported_objects, file_path_list

def rescale_objects(obj_list, scale_factor):
    rescaled_objects = []

    for obj in obj_list:
        obj.scale = (scale_factor, scale_factor, scale_factor)

        # world_matrix = obj.matrix_world.copy()
        # world_matrix = world_matrix.Scale(scale_factor, 4)  # 4 means scaling in 4 dimensions, for 3D objects
        # obj.matrix_world = world_matrix
        rescaled_objects.append(obj)
        # gt_trans_rots = gt_trans_rots * scale_factor
        # pred_trans_rots = pred_trans_rots * scale_factor
    
    # return rescaled_objects, gt_trans_rots, pred_trans_rots
    return rescaled_objects

def rotate_object_around_world_origin(obj, x_deg, y_deg, z_deg, reverse):
    if obj.type != 'MESH':
        return

    x_rad = math.radians(x_deg)
    y_rad = math.radians(y_deg)
    z_rad = math.radians(z_deg)

    loc = obj.location.copy()

    obj.location = Vector((0, 0, 0))
    bpy.context.view_layer.update()

    rot_x = Matrix.Rotation(x_rad, 4, 'X')
    rot_y = Matrix.Rotation(y_rad, 4, 'Y')
    rot_z = Matrix.Rotation(z_rad, 4, 'Z')

    if not reverse:
        rotation_matrix = rot_z @ rot_y @ rot_x
    else:
        rotation_matrix = rot_x @ rot_y @ rot_z


    obj.matrix_world = rotation_matrix @ obj.matrix_world
    obj.location = rotation_matrix @ loc

    # print(f"{obj.name} rotated.")

def rotate_all_meshes(obj_list, rotate_info, reverse = False):
    local_rotate_info = rotate_info.copy()
    x_deg = local_rotate_info[0]
    y_deg = local_rotate_info[1]
    z_deg = local_rotate_info[2]
    if reverse:
        x_deg *= -1
        y_deg *= -1
        z_deg *= -1 
    for obj in obj_list:
        rotate_object_around_world_origin(obj, x_deg, y_deg, z_deg, reverse)


def set_origin_to_geometry(obj):
    bpy.ops.object.select_all(action='DESELECT')  
    obj.select_set(True)  
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_MASS', center='BOUNDS')

def adjust_geometry_center(obj_list):
    total_weighted_center = Vector((0.0, 0.0, 0.0))
    total_volume = 0.0
    
    for obj in obj_list:
        set_origin_to_geometry(obj)
        
        if obj.bound_box:
            min_corner = Vector(obj.bound_box[0])
            max_corner = Vector(obj.bound_box[6])
            dimensions = max_corner - min_corner
            volume = dimensions.x * dimensions.y * dimensions.z
            
            center = obj.location
            total_weighted_center += center * volume
            total_volume += volume
    
    if total_volume > 0:
        weighted_average_center = total_weighted_center / total_volume
        
        for obj in obj_list:
            obj.location -= weighted_average_center    

def rotate_objects_z(objects, degree=1.0):
    """Rotate the given objects about the global Z axis by the specified degree."""
    angle_rad = math.radians(degree)
    Rz = Matrix.Rotation(angle_rad, 4, 'Z')
    for obj in objects:
        obj.matrix_world = Rz @ obj.matrix_world
    bpy.context.view_layer.update()


def import_obj_files(folder_path, model_folder_path, force_painting, redundant_path = None, removal_name = None, order_str = None,):
    """Manually import all OBJ files from a given folder."""
    # read all obj as a list
    if order_str:
        order_list = order_str.split(",")
        obj_files_all = [ f for f in order_list]
    else:
        obj_files_all = sorted([f for f in os.listdir(folder_path) if (f.endswith('.obj')or f.endswith('.ply'))], key=lambda x: int(x.split('_')[-1].split('.')[0]))

    # remove missing part
    l1 = len(obj_files_all)
    if removal_name:
        removal_obj_names = removal_name.split(',')
        if len(removal_obj_names[0].split('.'))>1:  
            obj_files = [name for name in obj_files_all if name.split('.')[-2] not in removal_obj_names]
        else:
            obj_files = [name for name in obj_files_all if name not in removal_obj_names]
    else:
        obj_files = obj_files_all
    l2 = len(obj_files)
    print("len before removal: ",l1," len after removal: ",l2)

    imported_objects = []

    origin_num = len(obj_files)
    imported_objects, total_file_path_list = create_final_object_list(obj_files, force_painting, folder_path)


    if redundant_path:
        redundant_list = redundant_path.split(',')
        imported_redundant_object, file_path_list_rdd = create_final_object_list(redundant_list, force_painting, model_folder_path)
        imported_objects = imported_objects + imported_redundant_object
        total_file_path_list = total_file_path_list + file_path_list_rdd

    return imported_objects, origin_num, total_file_path_list

def compute_final_transformation(init_pose, gt_transformation, transformation):
    """Compute the final transformation matrix based on the initial pose, ground truth, and current transformation."""
    trans1 = Vector(-init_pose[:3])
    trans_mat1 = Matrix.Translation(trans1)
    rot_mat1 = Quaternion(init_pose[3:]).inverted().to_matrix().to_4x4()

    trans2 = Vector(-gt_transformation[:3])
    trans_mat2 = Matrix.Translation(trans2)
    rot_mat2 = Quaternion(gt_transformation[3:]).inverted().to_matrix().to_4x4()

    trans3 = Vector(transformation[:3])
    trans_mat3 = Matrix.Translation(trans3)
    rot_mat3 = Quaternion(transformation[3:]).normalized().to_matrix().to_4x4()

    trans4 = Vector(init_pose[:3])
    trans_mat4 = Matrix.Translation(trans4)
    rot_mat4 = Quaternion(init_pose[3:]).to_matrix().to_4x4()

    final_transformation = rot_mat4 @ trans_mat4 @ trans_mat3 @ rot_mat3 @ rot_mat2 @ trans_mat2 @ trans_mat1 @ rot_mat1

    return final_transformation

def get_local_center_of_mass(obj):
    mesh = obj.data
    com = Vector((0.0, 0.0, 0.0))
    
    if mesh and len(mesh.vertices) > 0:
        for v in mesh.vertices:
            com += v.co
        com /= len(mesh.vertices)
        
    return com

def get_local_com_list(objects):
    """
    Given a list of Blender objects, return a list of local COM for each object.
    Order of local COMs corresponds to the order of `objects`.
    """
    local_com_list = []
    for obj in objects:
        local_com_list.append(get_local_center_of_mass(obj))
    return local_com_list

def apply_final_transformation(obj, init_pose, gt_transformation, transformation, com_list, obj_index):
    """Apply the computed final transformation to an object."""
    final_transformation = compute_final_transformation(init_pose, gt_transformation, transformation)
    obj.location = final_transformation.to_translation()
    obj.rotation_mode = 'QUATERNION'
    obj.rotation_quaternion = final_transformation.to_quaternion()
    
    bpy.context.view_layer.update()
    local_com = com_list[obj_index]
    return obj.matrix_world @ local_com

def interpolate_affine_numpy7(A, B, steps):
    if A.shape != (7,) or B.shape != (7,):
        raise ValueError("Both A and B must be numpy arrays of shape (7,)")
    
    transA = np.array(A[:3])
    transB = np.array(B[:3])
    
    quatA = Quaternion(A[3:7])
    quatB = Quaternion(B[3:7])
    
    result = []
    for i in range(1, steps + 1):
        t = i / float(steps)
        
        trans_interp = (1 - t) * transA + t * transB

        quat_interp = Quaternion.slerp(quatA, quatB, t)

        interp_array = np.concatenate([trans_interp, np.array([quat_interp.w, quat_interp.x, quat_interp.y, quat_interp.z])])
        result.append(interp_array)
    
    return result


def setup_light():
    """Add multiple lights to the scene."""
    def point_light_at(light, target_location):
        direction = Vector(target_location) - light.location
        light.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()

    # Add the first light
    bpy.ops.object.light_add(type='AREA', location=(2.7, -3, 2.7))
    light1 = bpy.context.object
    light1.data.energy = 500  # 500W in Blender's energy units
    light1.data.color = (1, 1, 1)  # White
    light1.data.size = 10  # 10m size
    point_light_at(light1, (0, 0, 0))

    # Add the second light
    bpy.ops.object.light_add(type='AREA', location=(-4.3, -3.4, 2.7))
    light2 = bpy.context.object
    light2.data.energy = 50  # 50W in Blender's energy units
    light2.data.color = (1, 1, 1)  # White
    light2.data.size = 10  # 10m size
    point_light_at(light2, (0, 0, 0))

    # Add the third light
    bpy.ops.object.light_add(type='AREA', location=(-1.8, 6, 0))
    light3 = bpy.context.object
    light3.data.energy = 1000  # 1000W in Blender's energy units
    light3.data.color = (1, 1, 1)  # White
    light3.data.size = 10  # 10m size
    point_light_at(light3, (0, 0, 0))

def setup_light_pre1():
    """Add multiple lights to the scene."""
    def point_light_at(light, target_location):
        direction = Vector(target_location) - light.location
        light.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()

    # Add the first light
    bpy.ops.object.light_add(type='AREA', location=(2.7, -3, 2.7))
    light1 = bpy.context.object
    light1.data.energy = 800  # 500W in Blender's energy units
    light1.data.color = (1, 1, 1)  # White
    light1.data.size = 10  # 10m size
    point_light_at(light1, (0, 0, 0))

    # Add the second light
    bpy.ops.object.light_add(type='AREA', location=(-4.3, -3.4, 2.7))
    light2 = bpy.context.object
    light2.data.energy = 300  # 50W in Blender's energy units
    light2.data.color = (1, 1, 1)  # White
    light2.data.size = 10  # 10m size
    point_light_at(light2, (0, 0, 0))

    # Add the third light
    bpy.ops.object.light_add(type='SPOT', location=(-3.31, 3.6, 4.38))
    light3 = bpy.context.object
    light3.data.energy = 1000  # 1000W in Blender's energy units
    light3.data.color = (1, 1, 1)  # White
    light3.data.shadow_soft_size = 2     
    light3.data.spot_size = math.radians(60)  
    light3.data.spot_blend = 1 
    point_light_at(light3, (0, 0, 0))

def setup_light_pre2():
    """Add multiple lights to the scene."""
    def point_light_at(light, target_location):
        direction = Vector(target_location) - light.location
        light.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()

    # Add the first light
    bpy.ops.object.light_add(type='SPOT', location=(2.83, -3, 4.01))
    light1 = bpy.context.object
    light1.data.energy = 1200  # 500W in Blender's energy units
    light1.data.color = (1, 1, 1)  # White
    light1.data.shadow_soft_size = 0.6  
    light1.data.spot_size = math.radians(70)  
    light1.data.spot_blend = 1     
    point_light_at(light1, (0, 0, 0))

    # Add the second light
    bpy.ops.object.light_add(type='AREA', location=(-4.3, -3.4, 2.7))
    light2 = bpy.context.object
    light2.data.energy = 300  # 50W in Blender's energy units
    light2.data.color = (1, 1, 1)  # White
    light2.data.size = 10  # 10m size
    point_light_at(light2, (0, 0, 0))

    # Add the third light
    bpy.ops.object.light_add(type='POINT', location=(-3.31, 3.6, 4.38))
    light3 = bpy.context.object
    light3.data.energy = 1000  # 1000W in Blender's energy units
    light3.data.color = (1, 1, 1)  # White
    light3.data.shadow_soft_size = 2     
    point_light_at(light3, (0, 0, 0))



def setup_camera(target):
    """Add and position a camera to focus on the target object."""
    bpy.ops.object.camera_add(location=(0, -4, 1.5))
    camera = bpy.context.object
    bpy.context.scene.camera = camera

    # Point the camera at the target
    direction = target.location - camera.location
    camera.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()

def set_white_background():
    """Set the render background to white with transparency and composite settings."""
    # Enable transparency in render settings
    bpy.context.scene.render.film_transparent = True

    # Set color management view transform to Standard
    bpy.context.scene.view_settings.view_transform = 'Standard'

    # Set up compositing
    bpy.context.scene.use_nodes = True
    comp_nodes = bpy.context.scene.node_tree.nodes
    comp_links = bpy.context.scene.node_tree.links

    # Clear existing nodes in the compositor
    for node in comp_nodes:
        comp_nodes.remove(node)

    # Add new nodes
    render_layers = comp_nodes.new(type="CompositorNodeRLayers")
    alpha_over = comp_nodes.new(type="CompositorNodeAlphaOver")
    composite_output = comp_nodes.new(type="CompositorNodeComposite")
    white_color = comp_nodes.new(type="CompositorNodeRGB")

    # Set the white color
    white_color.outputs[0].default_value = (1, 1, 1, 1)

    # Arrange nodes
    render_layers.location = (0, 0)
    white_color.location = (0, -200)
    alpha_over.location = (200, 0)
    composite_output.location = (400, 0)

    # Link nodes
    comp_links.new(render_layers.outputs[0], alpha_over.inputs[2])  # View layer to lower layer
    comp_links.new(white_color.outputs[0], alpha_over.inputs[1])   # White to upper layer
    comp_links.new(alpha_over.outputs[0], composite_output.inputs[0])  # Alpha over to composite

def set_pre_background():
    bpy.ops.mesh.primitive_plane_add(
        size=15,
        location=(0, 4, 0),
        rotation=(math.radians(90), 0, 0)
    )
    bpy.ops.mesh.primitive_plane_add(
        size=20,
        # location=(0, 0, -0.61)
        location=(0, 0, -1.5)

    )
    # bpy.ops.object.select_all(action='DESELECT')
    # second_bg = bpy.context.object
    # return second_bg 

# def move_plane(second_bg, init_height, target_height, frac):
#     new_height = (target_height - init_height)* frac + init_height
#     second_bg.location.z = new_height



def render_and_export(output_path, render = "EEVEE", fast_mode = True):
    """Render the scene and export the image."""
    if render == "EEVEE":
        bpy.context.scene.render.engine = 'BLENDER_EEVEE_NEXT'  # Use Cycles rendering engine
        # Eevee does not use device settings like Cycles  # Use CPU rendering to avoid Metal issues

        bpy.context.scene.cycles.samples = 1024  
    
    elif render == "EEVEE_GPU":
        bpy.context.scene.render.engine = 'BLENDER_EEVEE_NEXT'  # Use Cycles rendering engine

        bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
        bpy.context.preferences.addons['cycles'].preferences.get_devices()
        for device in bpy.context.preferences.addons['cycles'].preferences.devices:
            if device.type == 'CUDA':
                device.use = True
        bpy.context.scene.cycles.device = 'GPU'
        bpy.context.scene.cycles.samples = 1024  


    elif render == "CYCLES":
        bpy.context.scene.render.engine = 'CYCLES'
        bpy.context.scene.cycles.device = 'CPU'
        bpy.context.scene.cycles.samples = 1024
    elif render == "CYCLES_GPU":
        bpy.context.scene.render.engine = 'CYCLES'
        bpy.context.scene.cycles.device = 'GPU'
        bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'  # or 'OPTIX', 'OPENCL', 'METAL'
        bpy.context.scene.cycles.samples = 1024

        for device in bpy.context.preferences.addons['cycles'].preferences.devices:
            if device.type in {'CUDA', 'OPTIX', 'OPENCL', 'METAL'}:
                device.use = True

    else:
        print("must specify a render type")


    resolution = bpy.context.scene.render.resolution_y  # Use current width as the base
    bpy.context.scene.render.resolution_x = resolution  # Set height equal to width for square output
    if fast_mode:
        bpy.context.scene.render.resolution_percentage = 25
    else:
        bpy.context.scene.render.resolution_percentage = 100

    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)

# def save_blend_file(filepath):
#     """Save the current Blender file."""
#     bpy.ops.wm.save_as_mainfile(filepath=filepath)

def save_video(imgs_path, video_path, frame, inter_num = 1, length_t = 8):    
    # Compile frames into a video using FFmpeg
    command = [
        'ffmpeg', 
        '-framerate', f'{frame * 0.6 / (length_t/inter_num)}',  
        '-i', f'{imgs_path}/%04d.png',  # Adjust the pattern based on how your frames are named
        '-vf', 'tpad=stop_mode=clone:stop_duration=1',  # Hold the last frame for 2 seconds
        '-c:v', 'libx264', 
        '-pix_fmt', 'yuv420p', 
        '-crf', '17',  # Adjust CRF (lower means higher quality)
        video_path
    ]
    subprocess.run(command, check=True)

    print(f"Video saved to {video_path}")

def make_new_folder(output_folder, trans_path, model_path_half, rename):
    file_name = os.path.splitext(os.path.basename(trans_path))[0]
    if rename:
        file_name =file_name + "-" + model_path_half.replace("/", "-")

    # Create the new folder path
    new_folder_path = os.path.join(output_folder, "raw_data", file_name)

    # Create the folder
    os.makedirs(new_folder_path, exist_ok=True)
    print(f"Folder created: {new_folder_path}")

    return new_folder_path, file_name


def create_uv_sphere(location, obj ,radius):
    name = f"{obj.name}_vtx"
    # Ensure we're in Object Mode
    if bpy.context.object and bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')

    # Add a UV Sphere
    bpy.ops.mesh.primitive_uv_sphere_add(
        radius=radius,
        location=location
    )

    # Get the newly created sphere
    sphere = bpy.context.active_object
    sphere.name = name

    mat = bpy.data.materials.new(name=f"Material_dotline_{sphere.name}")
    obj_mat = obj.active_material
    mat.diffuse_color = obj_mat.node_tree.nodes.get("Principled BSDF").inputs[0].default_value
    sphere.data.materials.append(mat)

    new_collection = bpy.data.collections.get(name)
    if new_collection is None:
        new_collection = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(new_collection)

    for collection in sphere.users_collection:
        collection.objects.unlink(sphere)
    new_collection.objects.link(sphere)



def add_dotted_line(curve_obj, obj):
    geo_mod = None
    for mod in curve_obj.modifiers:
        if mod.type == 'NODES':
            # geo_mod = mod
            return
    if geo_mod == None:
        geo_mod = curve_obj.modifiers.new(name="MyGeoNodesModifier", type='NODES')
    
    # Create a new Geometry Node tree
    node_group = bpy.data.node_groups.new(name="MyGeoNodeTree", type="GeometryNodeTree")
    geo_mod.node_group = node_group

    # Clear existing nodes
    node_group.nodes.clear()

    # Create group input/output nodes
    group_input = node_group.nodes.new("NodeGroupInput")
    group_output = node_group.nodes.new("NodeGroupOutput")

    geo_in = node_group.interface.new_socket(socket_type="NodeSocketGeometry",name="Geometry", in_out = "INPUT")
    geo_out = node_group.interface.new_socket(socket_type="NodeSocketGeometry",name="Geometry", in_out = "OUTPUT")

    resample_node = node_group.nodes.new("GeometryNodeResampleCurve")
    resample_node.location = (0, 0)
    resample_node.mode = 'LENGTH'
    resample_node.inputs["Length"].default_value = 0.02
    
    ico_shpere = node_group.nodes.new("GeometryNodeMeshIcoSphere")
    ico_shpere.inputs["Radius"].default_value = 0.006
    
    instance_on_points = node_group.nodes.new("GeometryNodeInstanceOnPoints")
    
    set_material = node_group.nodes.new("GeometryNodeSetMaterial")
    
    node_group.links.new(group_input.outputs["Geometry"], resample_node.inputs["Curve"])
    node_group.links.new(resample_node.outputs["Curve"], instance_on_points.inputs["Points"])
    node_group.links.new(instance_on_points.outputs["Instances"], set_material.inputs["Geometry"])
    node_group.links.new(set_material.outputs["Geometry"], group_output.inputs["Geometry"])
    node_group.links.new(ico_shpere.outputs["Mesh"],instance_on_points.inputs["Instance"])
    
    mat = bpy.data.materials.new(name=f"Material_dotline_{curve_obj.name}")
    obj_mat = obj.active_material
    mat.diffuse_color = obj_mat.node_tree.nodes.get("Principled BSDF").inputs[0].default_value
    set_material.inputs["Material"].default_value = mat



    
def add_trajectory(imported_objects, objects_location_com):
    for obj_index, obj in enumerate(imported_objects):
        coords = [loc for loc in objects_location_com[obj_index]]
        
        if len(coords) < 2:
            continue
        
        # check if "Trajectory_xxx" exists
        traj_name = f"Trajectory_{obj.name}"
        existing_curve_obj = bpy.data.objects.get(traj_name)
        
        if existing_curve_obj and existing_curve_obj.type == 'CURVE':
            curve_data = existing_curve_obj.data
            curve_data.splines.clear()
            
            curve_data.dimensions = '3D'
            # curve_data.bevel_depth = 0.01
            
        else:
            curve_data = bpy.data.curves.new(name=f"TrajectoryCurve_{obj.name}", type='CURVE')
            curve_data.dimensions = '3D'
            # curve_data.bevel_depth = 0.01
            
            existing_curve_obj = bpy.data.objects.new(traj_name, curve_data)
            bpy.context.collection.objects.link(existing_curve_obj)
        add_dotted_line(existing_curve_obj, obj)
        poly = curve_data.splines.new('POLY')
        poly.points.add(len(coords) - 1)  
        
        for i, co in enumerate(coords):
            poly.points[i].co = (co.x, co.y, co.z, 1.0)

# def check_folder(address, max_retries = 10, wait_time = 0.5):
#     print(address)
#     retries = 0
#     while not os.path.exists(address) and retries < max_retries:
#         print(f"Folder {address} does not exist yet. Retrying...")
#         time.sleep(wait_time)  # Wait for a short time before retrying
#         retries += 1

def save_blend_file(blend_file_path, max_retries=1000, wait_time=0.5):
    
    retries = 0
    while retries < max_retries:
        try:
            bpy.ops.wm.save_as_mainfile(filepath=blend_file_path)
            print(f"File saved successfully: {blend_file_path}")
            return  
        except RuntimeError as e:
            print(f"Error saving file {blend_file_path}: {e}")
            print(f"Retrying... ({retries + 1}/{max_retries})")
            retries += 1
            time.sleep(wait_time)
    
    print(f"Failed to save file {blend_file_path} after {max_retries} retries.")
    raise RuntimeError(f"Failed to save the file after {max_retries} retries")

def sort_by_name(total_file_path_list, imported_objects, gt_trans_rots, pred_trans_rots,  origin_num):
    # sort the objects their corresponding trans by name

    # only change the main body order, not the redundant
    total_file_path_list_t = total_file_path_list[:origin_num].copy()
    imported_objects_t = imported_objects[:origin_num].copy()
    gt_trans_rots_t = gt_trans_rots[:origin_num].copy()
    pred_trans_rots_t = []
    for item in pred_trans_rots:
        pred_trans_rots_t.append(item[:origin_num].copy())


    order = sorted(range(len(total_file_path_list_t)), key=lambda i: total_file_path_list_t[i].split("/")[-1])
    total_file_path_list[:origin_num] = [total_file_path_list_t[i] for i in order]
    imported_objects[:origin_num] = [imported_objects_t[i] for i in order]
    gt_trans_rots[:origin_num] = [gt_trans_rots_t[i] for i in order]

    for idx, item in enumerate(pred_trans_rots_t):
        pred_trans_rots[idx][:origin_num] = [pred_trans_rots_t[idx][i] for i in order]


    return total_file_path_list, imported_objects, gt_trans_rots, pred_trans_rots


def generate(trans_path, output_folder, model_folder_path, dotted_line, data_mode, clean_mode, gt_mode, preview_mode, preview_rotate, rename, force_painting, first_texture_match, min_num, presentation, rotate_degree):
    """Main function to orchestrate the process."""
    if check_empty(trans_path, data_mode):
        return
    gt_trans_rots, pred_trans_rots, init_pose, model_path_half, redundant_path, removal_name, order_str, scale_factor = read_trans(trans_path, data_mode, gt_mode)
    if len(gt_trans_rots)<min_num:
        return
    output_folder_sub, trans_name = make_new_folder(output_folder, trans_path, model_path_half, rename)


    if data_mode == "jigsaw":
        pred_trans_rots = pred_trans_rots[None, :]
    model_path = os.path.join(model_folder_path, model_path_half)

    output_folder_video = os.path.join(output_folder, "video")
    output_folder_preview = os.path.join(output_folder, "preview")
    if not preview_mode or preview_rotate:
        os.makedirs(output_folder_video, exist_ok=True)
    os.makedirs(output_folder_preview, exist_ok=True)
    if data_mode != "puzzlefusion":
        shutil.copy2(trans_path, os.path.join(output_folder_sub,trans_name+".json"))
    else:
        shutil.copytree(trans_path, output_folder_sub, dirs_exist_ok=True)

    add_asset("material/pigeon_pastal.blend")

    reset_scene()

    # Delete default objects
    delete_default_objects()

    # Import all OBJ files
    imported_objects, origin_num, total_file_path_list = import_obj_files(model_path, model_folder_path, force_painting, redundant_path, removal_name, order_str)

    total_file_path_list, imported_objects, gt_trans_rots, pred_trans_rots = sort_by_name(total_file_path_list, imported_objects, gt_trans_rots, pred_trans_rots,  origin_num)

    imported_objects = rescale_objects(imported_objects, scale_factor)


    # # Assign random colors to each object
    for idx, obj in enumerate(imported_objects):
        # assign_random_color(obj)
        if idx < origin_num:
            adjust_material(obj, total_file_path_list[idx], "Pigeon Blue pastel SWISS KRONO plastic", False, True, force_painting, first_texture_match)
        else:
            adjust_material(obj, total_file_path_list[idx], "Pigeon Blue pastel SWISS KRONO plastic", False, False, force_painting, first_texture_match)
    # Set up light
    if not presentation:
        setup_light()
    else:
        setup_light_pre2()

    # Set up camera
    if imported_objects:
        setup_camera(imported_objects[0])  # Focus on the first imported object

    # Set background to white
    if not presentation:
        set_white_background()
    else:
        set_pre_background()
        # second_bg = set_pre_background()
        # init_height = second_bg.location.z


    frame =0
    num_obj = pred_trans_rots.shape[1]
    objects_location_com = [[]for _ in range(num_obj)]
    com_list = get_local_com_list(imported_objects)

    # gt_transform_record = [np.zeros(7) for i in range(len(imported_objects))]
    pred_transform_record = [np.zeros(7) for i in range(len(imported_objects))]   
    interpolate = False
    # Apply predicted transformations step by step and render
    for step_idx, step_transforms in enumerate(pred_trans_rots):
        if not presentation:
            for obj_index, (obj, gt_transform, pred_transform) in enumerate(
                zip(imported_objects, gt_trans_rots, step_transforms)
            ):
                location_com = apply_final_transformation(obj,  init_pose, gt_transform, pred_transform, com_list, obj_index)
                objects_location_com[obj_index].append(location_com)
                if (len(objects_location_com[obj_index])>1) and dotted_line:
                    create_uv_sphere(objects_location_com[obj_index][-2], obj, 0.01) # only plot the last position every time
                
                if gt_mode:
                    rotation_matrix = Matrix.Rotation(math.radians(180), 4, 'X')
                    obj.matrix_world @= rotation_matrix
            if dotted_line:
                add_trajectory(imported_objects, objects_location_com)
            step_output_path = os.path.join(output_folder_sub, f"{step_idx:04d}.png")
            if (not preview_mode) and (not presentation):
                render_and_export(step_output_path, "EEVEE", fast_mode = True)
            frame += 1
        elif presentation and (step_idx == 0):
            for obj_index, (obj, gt_transform, pred_transform) in enumerate(
                zip(imported_objects, gt_trans_rots, step_transforms)):
                location_com = apply_final_transformation(obj,  init_pose, gt_transform, pred_transform, com_list, obj_index)

                # gt_transform_record[obj_index] = gt_transform
                pred_transform_record[obj_index] = pred_transform

            step_output_path = os.path.join(output_folder_sub, f"{frame:04d}.png")
            rotate_all_meshes(imported_objects, rotate_degree, False)
            render_and_export(step_output_path, "EEVEE_GPU", fast_mode = False)
            rotate_all_meshes(imported_objects, rotate_degree, True)

            frame += 1
        
        elif presentation:
            # if (step_idx == 1):
            #     inter_num = 20
            # else:
            #     inter_num = 3
            inter_num = 5
            # gt_transform_new = [[] for i in range(len(imported_objects))]
            pred_transform_new = [[] for i in range(len(imported_objects))]                
            for inter_index in range(inter_num):
                for obj_index, (obj, gt_transform, pred_transform) in enumerate(
                    zip(imported_objects, gt_trans_rots, step_transforms)):
                    if step_idx != 0 and inter_index == 0:
                        # gt_transform_new[obj_index] = interpolate_affine_numpy7(gt_transform_record[obj_index], gt_transform, inter_num)
                        pred_transform_new[obj_index] = interpolate_affine_numpy7(pred_transform_record[obj_index], pred_transform, inter_num)
                    location_com = apply_final_transformation(obj,  init_pose, gt_transform, pred_transform_new[obj_index][inter_index], com_list, obj_index)

                    if inter_index == (inter_num-1):
                        # gt_transform_record[obj_index] = gt_transform
                        pred_transform_record[obj_index] = pred_transform

                step_output_path = os.path.join(output_folder_sub, f"{frame:04d}.png")
                rotate_all_meshes(imported_objects, rotate_degree, False)
                render_and_export(step_output_path, "EEVEE_GPU", fast_mode = False)
                rotate_all_meshes(imported_objects, rotate_degree, True)
                frame += 1








    preview_output_path = os.path.join(output_folder_preview, f"{trans_name}.png")
    if preview_mode:
        adjust_geometry_center(imported_objects)
    render_and_export(preview_output_path, "EEVEE", fast_mode = False)
    if not preview_mode and not presentation:
        save_video(imgs_path = output_folder_sub, video_path = output_folder_video+ f"/{trans_name}.mp4", frame= frame)
    elif preview_rotate:
        rotate = 20
        degree = 360/rotate
        for rot_i in range(rotate):
            rotate_objects_z(imported_objects, degree)
            step_output_path = os.path.join(output_folder_sub, f"{rot_i+1:04d}.png")
            render_and_export(step_output_path, "EEVEE", fast_mode = False)
        save_video(imgs_path = output_folder_sub, video_path = output_folder_video+ f"/{trans_name}.mp4", frame= rotate)
    elif presentation:
        rotate_all_meshes(imported_objects, rotate_degree, False)
        # target_height = -0.61
        height_steps = 15
        for i in range(height_steps):
            # move_plane(second_bg, init_height, target_height, (i+1)/height_steps)


            step_output_path = os.path.join(output_folder_sub, f"{frame:04d}.png")
            render_and_export(step_output_path, "EEVEE_GPU", fast_mode = False)
            frame += 1

        rotate = 100
        degree = 360/rotate
        for rot_i in range(rotate):
            rotate_objects_z(imported_objects, degree)
            step_output_path = os.path.join(output_folder_sub, f"{frame:04d}.png")
            render_and_export(step_output_path, "EEVEE_GPU", fast_mode = False)
            frame += 1
        save_video(imgs_path = output_folder_sub, video_path = output_folder_video+ f"/{trans_name}.mp4", frame= frame, inter_num = inter_num)



    fill_colors()




    # Save the Blender file
    if not clean_mode:
        blend_file_path = os.path.join(output_folder_sub, f"{trans_name}.blend")
        save_blend_file(blend_file_path)
    else:
        # clean png data
        png_files = glob.glob(os.path.join(output_folder_sub, "*.png"))
        for png_file in png_files:
            os.remove(png_file)


def add_asset(source_blend_path):
    if not os.path.exists(source_blend_path):
        print(f"source .blend file does not exist: {source_blend_path}")
        return

    with bpy.data.libraries.load(source_blend_path, link=False) as (data_from, data_to):
        for mat in data_from.materials:
            if mat not in data_to.materials:
                data_to.materials.append(mat)
        


def adjust_material(obj, obj_path, material_name, use_asset, main_part, force_painting, first_texture_match):
    # if texture is provided, use texture.
    color_attr_names = [attr.name for attr in obj.data.color_attributes] # select "Col"

    if not force_painting:
        # if first_texture_match:
        #     if match_texture:
        #         match_texture
        #         return
        #     elif "Col":
        #             use "Col"
        #             return
        # else:
        #     if "Col":
        #         use "Col"
        #     elif match_texture:
        #         match_texture
        #         return

        if first_texture_match:
            if obj_path.lower().endswith(".obj"):
                jpg_path = obj_path.replace("obj", "jpg")
            else:
                jpg_path = obj_path.replace("ply", "jpg")
            if os.path.exists(jpg_path):
                mat = bpy.data.materials.new(name="Texture_Material")
                mat.use_nodes = True
                nodes = mat.node_tree.nodes
                links = mat.node_tree.links

                texture_image = bpy.data.images.load(jpg_path)
                
                texture_node = nodes.new(type="ShaderNodeTexImage")
                texture_node.image = texture_image
                texture_node.name = "Image Texture"
                
                principled_bsdf = nodes.get("Principled BSDF")
                links.new(texture_node.outputs["Color"], principled_bsdf.inputs["Base Color"])

                if obj.data.materials:
                    obj.data.materials[0] = mat
                else:
                    obj.data.materials.append(mat)
                return
            elif "Col" in color_attr_names:
                mat = bpy.data.materials.new(name="NewMaterialWithVertexColor")
                mat.use_nodes = True
                nodes = mat.node_tree.nodes
                links = mat.node_tree.links

                vc_node = nodes.new(type="ShaderNodeVertexColor")
                vc_node.name = "Color Attribute"
                vc_node.layer_name = "Col"

                principled_bsdf = nodes.get("Principled BSDF")
                links.new(vc_node.outputs["Color"], principled_bsdf.inputs["Base Color"])

                if obj.data.materials:
                    obj.data.materials[0] = mat
                else:
                    obj.data.materials.append(mat)
                return
        else:
            if obj_path.lower().endswith(".obj"):
                jpg_path = obj_path.replace("obj", "jpg")
            else:
                jpg_path = obj_path.replace("ply", "jpg")
            if "Col" in color_attr_names:
                mat = bpy.data.materials.new(name="NewMaterialWithVertexColor")
                mat.use_nodes = True
                nodes = mat.node_tree.nodes
                links = mat.node_tree.links

                vc_node = nodes.new(type="ShaderNodeVertexColor")
                vc_node.name = "Color Attribute"
                vc_node.layer_name = "Col"

                principled_bsdf = nodes.get("Principled BSDF")
                links.new(vc_node.outputs["Color"], principled_bsdf.inputs["Base Color"])

                if obj.data.materials:
                    obj.data.materials[0] = mat
                else:
                    obj.data.materials.append(mat)
                return
            elif os.path.exists(jpg_path):
                mat = bpy.data.materials.new(name="Texture_Material")
                mat.use_nodes = True
                nodes = mat.node_tree.nodes
                links = mat.node_tree.links

                texture_image = bpy.data.images.load(jpg_path)
                
                texture_node = nodes.new(type="ShaderNodeTexImage")
                texture_node.image = texture_image
                texture_node.name = "Image Texture"
                
                principled_bsdf = nodes.get("Principled BSDF")
                links.new(texture_node.outputs["Color"], principled_bsdf.inputs["Base Color"])

                if obj.data.materials:
                    obj.data.materials[0] = mat
                else:
                    obj.data.materials.append(mat)
                return

    # else, just apply color set.
    mat = None
    if use_asset:
        asset_mat = bpy.data.materials.get(material_name)
        mat = asset_mat.copy()
        if mat is None:
            print(f"Material '{material_name}' not found in current file!")
            return
        mat.name = f"Material_{obj.name}"
        if len(obj.data.materials) == 0:
            obj.data.materials.append(mat)
        else:
            obj.data.materials[0] = mat
    else:
        # if not using material from asset, create a new one.
        mat = obj.active_material
        if not mat:
            mat = bpy.data.materials.new(name=f"Material_{obj.name}")
            obj.data.materials.append(mat)

    # assign random color
    mat.use_nodes = True
    # Get or create Principled BSDF
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if not bsdf:
        bsdf = mat.node_tree.nodes.new(type="ShaderNodeBsdfPrincipled")

    # Set random color and link the BSDF to Material Output
    # bsdf.inputs[0].default_value = (random.random(), random.random(), random.random(), 1)  # RGBA
    if main_part:
        bsdf.inputs[0].default_value = random_color()  # RGBA
    bsdf.inputs[2].default_value = 0.35
    output = mat.node_tree.nodes.get("Material Output")
    if not output:
        output = mat.node_tree.nodes.new(type="ShaderNodeOutputMaterial")
    mat.node_tree.links.new(bsdf.outputs[0], output.inputs[0])




# Color management
def assign_random_color(obj):
    """Assign a random color to the given object."""
    mat = obj.active_material
    if not mat:
        mat = bpy.data.materials.new(name=f"Material_{obj.name}")
        obj.data.materials.append(mat)
    
    mat.use_nodes = True
    # Get or create Principled BSDF
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if not bsdf:
        bsdf = mat.node_tree.nodes.new(type="ShaderNodeBsdfPrincipled")

    # Set random color and link the BSDF to Material Output
    # bsdf.inputs[0].default_value = (random.random(), random.random(), random.random(), 1)  # RGBA
    bsdf.inputs[0].default_value = random_color()  # RGBA
    output = mat.node_tree.nodes.get("Material Output")
    if not output:
        output = mat.node_tree.nodes.new(type="ShaderNodeOutputMaterial")
    mat.node_tree.links.new(bsdf.outputs[0], output.inputs[0])

# https://color-term.com/lego-colors/
red = (201/255,26/255,9/255,1)
blue = (0/255,85/255,191/255,1)
green = (35/255,120/255,65/255,1)
salmon = (242/255, 112/255,94/255,1)
yellow = (252/255,151/255,172/255,1)
orange = (254/255,138/255,24/255,1)
bright_green = (75/255,159/255,74/255,1)
dark_turquoise = (0,143/255,155/255,1)
trans_yellow = (245/255,205/255,47/255,1)
violet = (67/255,84/255,163/255,1)
aqua = (179/255,215/255,209/255,1)
medium_lime = (199/255,210/255,60/255,1)
trans_neon_orange = (255/255,128/255,13/255,1)

colors = []
def fill_colors():
    global colors
    colors = [orange, red, green, blue, salmon, yellow, bright_green, dark_turquoise,trans_yellow,violet]
fill_colors()