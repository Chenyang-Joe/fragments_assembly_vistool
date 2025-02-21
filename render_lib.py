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
import h5py

def select_model(folder_path, file_num, single_mode, random_draw, traversal_all, range, data_mode):
    if single_mode:
        return [folder_path]
    if not os.path.isdir(folder_path):
        raise ValueError(f"{folder_path} not a avaliable folder path")

    if data_mode != "puzzlefusion":
        all_files = [os.path.join(folder_path, name) for name in os.listdir(folder_path)
                    if (os.path.isfile(os.path.join(folder_path, name)) and (name[0]!="."))]
        all_files = sorted(all_files, key=lambda x: int(os.path.basename(x).split('.')[0]))
    else:
        all_files = [os.path.join(folder_path, name) for name in os.listdir(folder_path)
            if os.path.isdir(os.path.join(folder_path, name))]
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
    redundant_path = ''
    removal_name = ''
    if data_mode != "puzzlefusion":
        with open(trans_file_path, 'r') as f:
            data = json.load(f)
        gt_trans_rots = np.array(data['gt_trans_rots'])
        pred_trans_rots = np.array(data['pred_trans_rots'])
        # init_pose = np.array(data['init_pose'])
        init_pose = np.array([0,0,0,0,0,0,0])
        model_path = data['name']
        if 'redundant_pieces' in data.keys():
            redundant_path = data['redundant_pieces']
        if 'removal_pieces' in data.keys():
            removal_name = data['removal_pieces']
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
            model_path = file.read().strip()

    if gt_mode:
        pred_trans_rots = pred_trans_rots[:1, :, :]*0
        gt_trans_rots = gt_trans_rots*0
        init_pose = init_pose*0



    return gt_trans_rots, pred_trans_rots, init_pose, model_path, redundant_path, removal_name

def reset_scene():
    """Reset the Blender scene by deleting all objects and clearing unused data."""
    # Delete all objects in the scene
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    # Clear unused data blocks
    bpy.ops.outliner.orphans_purge(do_recursive=True)


def delete_default_objects():
    """Delete default objects like Light, Camera, and Cube."""
    bpy.ops.object.select_all(action='DESELECT')
    for obj in bpy.data.objects:
        if obj.name in ['Light', 'Camera', 'Cube']:
            obj.select_set(True)
    bpy.ops.object.delete()

def parse_obj(filepath):
    """Parse an OBJ file and extract vertices and faces."""
    vertices = []
    faces = []

    with open(filepath, 'r') as file:
        for line in file:
            if line.startswith('v '):
                parts = line.split()
                vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif line.startswith('f '):
                parts = line.split()
                face = [int(p.split('/')[0]) - 1 for p in parts[1:]]
                faces.append(face)

    return vertices, faces

def import_obj_files(folder_path, model_folder_path, redundant_path = None, removal_name = None):
    """Manually import all OBJ files from a given folder."""
    obj_files_all = sorted([f for f in os.listdir(folder_path) if f.endswith('.obj')], key=lambda x: int(x.split('_')[-1].split('.')[0]))
    l1 = len(obj_files_all)
    if removal_name:
        removal_obj_names = removal_name.split(',')
        obj_files = [name for name in obj_files_all if name.split('.')[-2] not in removal_obj_names]
    else:
        obj_files = obj_files_all
    l2 = len(obj_files)
    print("len before removal: ",l1," len after removal: ",l2)

    imported_objects = []

    origin_num = len(obj_files)
    for obj_file in obj_files:
        file_path = os.path.join(folder_path, obj_file)
        vertices, faces = parse_obj(file_path)

        # Create a new mesh and object in Blender
        mesh = bpy.data.meshes.new(name=f"Mesh_{obj_file}")
        mesh.from_pydata(vertices, [], faces)
        mesh.update()

        obj = bpy.data.objects.new(name=f"Object_{obj_file}", object_data=mesh)
        bpy.context.collection.objects.link(obj)
        imported_objects.append(obj)

    if redundant_path:
        redundant_list = redundant_path.split(',')
        for obj_file in redundant_list:
            file_path = os.path.join(model_folder_path, obj_file[:-3]+'_'+obj_file[-1]+'.obj')
            vertices, faces = parse_obj(file_path)

            # Create a new mesh and object in Blender
            mesh = bpy.data.meshes.new(name=f"Mesh_{obj_file}")
            mesh.from_pydata(vertices, [], faces)
            mesh.update()

            obj = bpy.data.objects.new(name=f"Object_{obj_file}", object_data=mesh)
            bpy.context.collection.objects.link(obj)
            imported_objects.append(obj)

    return imported_objects, origin_num

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

def render_and_export(output_path, render = "EEVEE", fast_mode = True):
    """Render the scene and export the image."""
    if render == "EEVEE":
        bpy.context.scene.render.engine = 'BLENDER_EEVEE_NEXT'  # Use Cycles rendering engine
        # Eevee does not use device settings like Cycles  # Use CPU rendering to avoid Metal issues
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

def save_blend_file(filepath):
    """Save the current Blender file."""
    bpy.ops.wm.save_as_mainfile(filepath=filepath)

def save_video(imgs_path, video_path, frame):    
    # Compile frames into a video using FFmpeg
    command = [
        'ffmpeg', 
        '-framerate', f'{frame / 8}',  
        '-i', f'{imgs_path}/%04d.png',  # Adjust the pattern based on how your frames are named
        '-vf', 'tpad=stop_mode=clone:stop_duration=1',  # Hold the last frame for 2 seconds
        '-c:v', 'libx264', 
        '-pix_fmt', 'yuv420p', 
        '-crf', '17',  # Adjust CRF (lower means higher quality)
        video_path
    ]
    subprocess.run(command, check=True)

    print(f"Video saved to {video_path}")

def make_new_folder(output_folder, trans_path):
    file_name = os.path.splitext(os.path.basename(trans_path))[0]

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




def generate(trans_path, output_folder, model_folder_path, dotted_line, data_mode, clean_mode, gt_mode, preview_mode):
    """Main function to orchestrate the process."""
    if check_empty(trans_path, data_mode):
        return
    gt_trans_rots, pred_trans_rots, init_pose, model_path_half, redundant_path, removal_name = read_trans(trans_path, data_mode, gt_mode)

    if data_mode == "jigsaw":
        pred_trans_rots = pred_trans_rots[None, :]
    model_path = os.path.join(model_folder_path, model_path_half)
    output_folder_sub, trans_name = make_new_folder(output_folder, trans_path)
    output_folder_video = os.path.join(output_folder, "video")
    output_folder_preview = os.path.join(output_folder, "preview")
    if not preview_mode:
        os.makedirs(output_folder_video, exist_ok=True)
    os.makedirs(output_folder_preview, exist_ok=True)
    if data_mode != "puzzlefusion":
        shutil.copy2(trans_path, output_folder_sub)
    else:
        shutil.copytree(trans_path, output_folder_sub, dirs_exist_ok=True)

    add_asset("material/pigeon_pastal.blend")

    reset_scene()

    # Delete default objects
    delete_default_objects()

    # Import all OBJ files
    imported_objects, origin_num = import_obj_files(model_path, model_folder_path, redundant_path, removal_name)

    # # Assign random colors to each object
    for idx, obj in enumerate(imported_objects):
        # assign_random_color(obj)
        if idx < origin_num:
            adjust_material(obj, "Pigeon Blue pastel SWISS KRONO plastic", False, True)
        else:
            adjust_material(obj, "Pigeon Blue pastel SWISS KRONO plastic", False, False)
    # Set up light
    setup_light()

    # Set up camera
    if imported_objects:
        setup_camera(imported_objects[0])  # Focus on the first imported object

    # Set background to white
    set_white_background()

    frame =0
    num_obj = pred_trans_rots.shape[1]
    objects_location_com = [[]for _ in range(num_obj)]
    com_list = get_local_com_list(imported_objects)
    # Apply predicted transformations step by step and render
    for step_idx, step_transforms in enumerate(pred_trans_rots):
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
        if not preview_mode:
            render_and_export(step_output_path, "EEVEE", fast_mode = True)
        frame += 1
    preview_output_path = os.path.join(output_folder_preview, f"{trans_name}.png")
    render_and_export(preview_output_path, "EEVEE", fast_mode = False)
    fill_colors()
    if not preview_mode:
        save_video(imgs_path = output_folder_sub, video_path = output_folder_video+ f"/{trans_name}.mp4", frame= frame)

    # Save the Blender file
    if not clean_mode:
        blend_file_path = os.path.join(output_folder_sub, f"scene{trans_name}.blend")
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
        


def adjust_material(obj, material_name, use_asset, main_part):
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