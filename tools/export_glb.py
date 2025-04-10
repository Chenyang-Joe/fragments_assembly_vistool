import bpy
import sys
import os
import gc  # Garbage collector


def change_to_glb(blend_file):
    output_file = blend_file[:-6]+".glb"

    if not os.path.exists(blend_file):
        print("error, blend file does not exist" + blend_file)
        return

    bpy.ops.wm.open_mainfile(filepath=blend_file)
    print("oepn: " + blend_file)

    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            if "Plane" in obj.name:
                print("Excluding object: " + obj.name)
                obj.select_set(False)
            # Do not select this object.
            else:
            # Select objects that are not excluded.
                obj.select_set(True)



                bpy.context.view_layer.objects.active = obj  # Make it active so we can add modifiers

                # edit here, change generated UV map by default to the method you describe

                # Check number of faces
                num_faces = len(obj.data.polygons)
                print(f"{obj.name} has {num_faces} faces")

                if num_faces > 10000:
                    ratio = 10000.0 / num_faces
                    # Add decimate modifier
                    decimate = obj.modifiers.new(name="Decimate", type='DECIMATE')
                    decimate.ratio = ratio
                    decimate.use_collapse_triangulate = True
                    print(f"Applied decimation with ratio: {ratio}")
                else:
                    print(f"{obj.name} has less than 10000 faces, skipping decimation")



    # bpy.context.view_layer.update()


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
    bpy.ops.wm.read_factory_settings(use_empty=True)  # Reset Blender to empty scene
    bpy.data.orphans_purge(do_recursive=True)         # Remove unused data blocks
    gc.collect()                                       # Python-level garbage collection
    print("Memory cleanup done.\n")

outter_path = "/mnt/NAS/data/assembly/last_week/vis/video/pastlife_real_ca_vt"
raw_data = os.path.join(outter_path, "raw_data")

# Get a list of subdirectories within raw_data.
folder_list = [os.path.join(raw_data, d) for d in os.listdir(raw_data) if os.path.isdir(os.path.join(raw_data, d))]

folder_list = [folder_list[0]]

# Loop through each folder and find blend files.
for folder in folder_list:
    for f in os.listdir(folder):
        if f.endswith(".blend"):
            print(f)
            change_to_glb(os.path.join(folder,f))
