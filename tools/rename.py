"""
0. set output, generate new folder
1. read preview number, get list
2. get corresponding source filde address
3. copy, video, image into new folder

"""

import os
import json

# artifact = False
puzzlefusion = False
input = "/mnt/NAS/data/assembly/assembly_vis_final/jigsaw_logs_and_matching_data/artifact_vol_missing_4"
# output = "jigsaw_rename/everyday_vol"

# os.makedirs(os.path.join(output, "preview"), exist_ok=True)
# os.makedirs(os.path.join(output, "raw_data"), exist_ok=True)
# os.makedirs(os.path.join(output, "video"), exist_ok=True)

preview_path = os.path.join(input, "preview")
video_path = os.path.join(input, "video")
file_names = [f for f in os.listdir(preview_path) if os.path.isfile(os.path.join(preview_path, f))]
file_names = sorted(file_names, key=lambda x: int(x.split(".")[0]))

for name in file_names:
    number = name.split(".")[0]
    if not puzzlefusion:
        raw_file = os.path.join(input, "raw_data", number, number+".json")
        with open(raw_file, 'r') as f:
            json_data = json.load(f)
        raw_path = json_data["name"]
    else:
        raw_file = os.path.join(input, "raw_data", number, "mesh_file_path.txt")
        with open(raw_file,'r') as f:
            raw_path = f.read()
    new_name = number + "-" + raw_path.replace("/", "-")
    # name_list = raw_path.split("/")
    # if artifact:
    #     new_name = name_list[1]+"_"+name_list[2]+"_"+number
    # else:
    #     new_name = name_list[1]+"_"+name_list[2][:6]+"_"+name_list[3]+"_"+number
    os.rename(os.path.join(preview_path, name), os.path.join(preview_path, new_name+".png"))
    # os.rename(os.path.join(video_path, number+".mp4"), os.path.join(video_path, new_name+".mp4"))


