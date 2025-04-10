import bpy
import sys
import os
import gc  # Garbage collector

def apply_uv_projection_and_bake(obj):
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)

    # Ensure UV map exists
    if not obj.data.uv_layers:
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.uv.cube_project(cube_size=1.0)
        bpy.ops.object.mode_set(mode='OBJECT')

    # Create baked image
    baked_image = bpy.data.images.new(name="BakedTexture", width=1024, height=1024)

    # Create a new image texture node for baking
    mat = obj.active_material
    if not mat:
        mat = bpy.data.materials.new(name="BakedMaterial")
        obj.data.materials.append(mat)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    image_node = nodes.new('ShaderNodeTexImage')
    image_node.image = baked_image
    image_node.select = True
    mat.node_tree.nodes.active = image_node  # Needed for baking target

    # Set bake type and perform baking
    bpy.context.scene.render.engine = 'CYCLES'  # Ensure Cycles engine is used
    bpy.context.scene.cycles.bake_type = 'DIFFUSE'
    bpy.ops.object.bake(type='DIFFUSE', pass_filter={'COLOR'}, use_clear=True)

    print(f"Baked texture to image for object: {obj.name}")

def change_to_glb(blend_file):
    output_file = blend_file[:-6] + ".glb"

    if not os.path.exists(blend_file):
        print("error, blend file does not exist" + blend_file)
        return

    bpy.ops.wm.open_mainfile(filepath=blend_file)
    print("open: " + blend_file)

    # Ensure render engine is Cycles for baking
    bpy.context.scene.render.engine = 'CYCLES'

    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            if "Plane" in obj.name:
                print("Excluding object: " + obj.name)
                obj.select_set(False)
            else:
                obj.select_set(True)
                bpy.context.view_layer.objects.active = obj

                # Apply UV projection and bake
                apply_uv_projection_and_bake(obj)

                # Decimate if too many faces
                num_faces = len(obj.data.polygons)
                print(f"{obj.name} has {num_faces} faces")

                if num_faces > 10000:
                    ratio = 10000.0 / num_faces
                    decimate = obj.modifiers.new(name="Decimate", type='DECIMATE')
                    decimate.ratio = ratio
                    decimate.use_collapse_triangulate = True
                    print(f"Applied decimation with ratio: {ratio}")
                else:
                    print(f"{obj.name} has less than 10000 faces, skipping decimation")

    bpy.ops.export_scene.gltf(
        filepath=output_file,
        export_format='GLB',
        export_materials='EXPORT',
        export_apply=True,
        use_selection=True
    )
    print("success to export" + output_file)

    # Memory cleanup
    print("Cleaning up memory...")
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.data.orphans_purge(do_recursive=True)
    gc.collect()
    print("Memory cleanup done.\n")

outter_path = "/mnt/NAS/data/assembly/last_week/vis/video/pastlife_real_change_angle"
raw_data = os.path.join(outter_path, "raw_data")

folder_list = [os.path.join(raw_data, d) for d in os.listdir(raw_data) if os.path.isdir(os.path.join(raw_data, d))]
folder_list = [folder_list[0]]

for folder in folder_list:
    for f in os.listdir(folder):
        if f.endswith(".blend"):
            print(f)
            change_to_glb(os.path.join(folder, f))
