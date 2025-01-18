import bpy
import os
import random
import json
import subprocess
from mathutils import Quaternion, Vector, Matrix
import numpy as np

def select_model(folder_path, file_num, single_mode):
    if single_mode:
        return [folder_path]
    if not os.path.isdir(folder_path):
        raise ValueError(f"{folder_path} not a avaliable folder path")

    all_files = [os.path.join(folder_path, name) for name in os.listdir(folder_path)
                   if os.path.isfile(os.path.join(folder_path, name))]

    if file_num > len(all_files):
        raise ValueError("folder_num excesses the total folder number")

    file_list = random.sample(all_files, file_num)

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

def read_json(json_file_path):
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    gt_trans_rots = np.array(data['gt_trans_rots'])
    pred_trans_rots = np.array(data['pred_trans_rots'])
    # init_pose = np.array(data['init_pose'])
    init_pose = np.array([0,0,0,0,0,0,0])
    model_path = data['name']
    return gt_trans_rots, pred_trans_rots, init_pose, model_path

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

def import_obj_files(folder_path):
    """Manually import all OBJ files from a given folder."""
    obj_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.obj')], key=lambda x: int(x.split('_')[-1].split('.')[0]))
    imported_objects = []

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

    return imported_objects

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

def apply_final_transformation(obj, init_pose, gt_transformation, transformation):
    """Apply the computed final transformation to an object."""
    final_transformation = compute_final_transformation(init_pose, gt_transformation, transformation)
    obj.location = final_transformation.to_translation()
    obj.rotation_mode = 'QUATERNION'
    obj.rotation_quaternion = final_transformation.to_quaternion()

def adjust_material_to_metal(objects):
    """Adjust all objects' materials to have a plastic-like appearance."""
    for obj in objects:
        if obj.type == 'MESH':
            # Ensure the object has a material
            mat = obj.active_material
            if not mat:
                mat = bpy.data.materials.new(name=f"Material_{obj.name}")
                obj.data.materials.append(mat)

            mat.use_nodes = True
            nodes = mat.node_tree.nodes
            links = mat.node_tree.links

            # Clear existing nodes
            for node in nodes:
                nodes.remove(node)

            # Add Principled BSDF for plastic appearance
            bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
            bsdf.location = (0, 0)
            bsdf.inputs[4].default_value = 0  # Roughness for plastic
            bsdf.inputs[7].default_value = 0.1  # Metallic

            # Add Material Output node
            output = nodes.new(type="ShaderNodeOutputMaterial")
            output.location = (200, 0)

            # Link BSDF to Material Output
            links.new(bsdf.outputs[0], output.inputs[0])

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
    bpy.ops.object.camera_add(location=(0, -3, 1))
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

def render_and_export(output_path):
    """Render the scene and export the image."""
    bpy.context.scene.render.engine = 'BLENDER_EEVEE_NEXT'  # Use Cycles rendering engine
    # Eevee does not use device settings like Cycles  # Use CPU rendering to avoid Metal issues

    resolution = bpy.context.scene.render.resolution_y  # Use current width as the base
    bpy.context.scene.render.resolution_x = resolution  # Set height equal to width for square output


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

def make_new_folder(output_folder, json_path):
    file_name = os.path.splitext(os.path.basename(json_path))[0]

    # Create the new folder path
    new_folder_path = os.path.join(output_folder, file_name)

    # Create the folder
    os.makedirs(new_folder_path, exist_ok=True)
    print(f"Folder created: {new_folder_path}")

    return new_folder_path, file_name


def generate(json_path, output_folder, model_folder_path):
    """Main function to orchestrate the process."""
    gt_trans_rots, pred_trans_rots, init_pose, model_path = read_json(json_path)
    model_path = os.path.join(model_folder_path, model_path)
    output_folder_sub, json_name = make_new_folder(output_folder, json_path)
    output_folder_video = os.path.join(output_folder, "video")
    os.makedirs(output_folder_video, exist_ok=True)

    reset_scene()

    # Delete default objects
    delete_default_objects()

    # Import all OBJ files
    imported_objects = import_obj_files(model_path)

    # # Assign random colors to each object
    for obj in imported_objects:
        assign_random_color(obj)

    # Set up light
    setup_light()

    # Set up camera
    if imported_objects:
        setup_camera(imported_objects[0])  # Focus on the first imported object

    # Set background to white
    set_white_background()

    # # Apply ground truth transformations
    # for obj, gt_transform in zip(imported_objects, gt_trans_rots):
    #     apply_final_transformation(obj, init_pose, gt_transform, [0, 0, 0, 0, 0, 0, 0])

    # # Render initial state
    # initial_output_path = os.path.join(output_folder, "0000.png")
    # render_and_export(initial_output_path)

    frame =0
    # Apply predicted transformations step by step and render
    for step_idx, step_transforms in enumerate(pred_trans_rots):
        for obj, gt_transform, pred_transform in zip(
            imported_objects, gt_trans_rots, step_transforms
        ):
            apply_final_transformation(obj, init_pose, gt_transform, pred_transform)
        step_output_path = os.path.join(output_folder_sub, f"{step_idx:04d}.png")
        render_and_export(step_output_path)
        frame += 1

    fill_colors()
    save_video(imgs_path = output_folder_sub, video_path = output_folder_video+ f"/{json_name}.mp4", frame= frame)

    # Save the Blender file
    blend_file_path = os.path.join(output_folder_sub, "scene.blend")
    save_blend_file(blend_file_path)


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


def fill_colors():
    global colors
    colors = [orange, red, green, blue, salmon, yellow, bright_green, dark_turquoise,trans_yellow,violet]
fill_colors()

# Example usage
# json_folder_path = "/Users/chenyangxu/Library/Mobile Documents/com~apple~CloudDocs/School/NYU 2024 Winter/AI4CE/models/breaking-bad preprocessing/trans_samples_large_constrained/1234.json"
# model_folder_path = "/Users/chenyangxu/Library/Mobile Documents/com~apple~CloudDocs/School/NYU 2024 Winter/AI4CE/models/breaking-bad preprocessing"
# output_folder = "output"
# single_mode = True
# file_num = 100

json_folder_path = "your transformation json file/your folder with transformation json files"
model_folder_path = "your breaking bad folder"
output_folder = "your output folder"
single_mode = True
file_num = 100 

if __name__ == "__main__":
    file_list = select_model(json_folder_path, file_num, single_mode)
    for json_path in file_list:
        generate(json_path, output_folder, model_folder_path)

