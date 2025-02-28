import sys
import os
import json

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)

from render_lib import *


config_file = "config.json"
with open(config_file, "r") as file:
    config = json.load(file)

trans_folder_path = config["trans_folder_path"]
model_folder_path = config["model_folder_path"]
output_folder = config["output_folder"]
single_mode = config["single_mode"]
traversal_all = config["traversal_all"]
random_draw = config["random_draw"]
range = config["range"]
file_num = config["file_num"]
dotted_line = config["dotted_line"]
data_mode = config["data_mode"]
clean_mode = config["clean_mode"]
gt_mode = config["gt_mode"]
preview_mode = config["preview_mode"]
rename = config["rename"]


if __name__ == "__main__":
    file_list = select_model(trans_folder_path, file_num, single_mode, random_draw, traversal_all, range, data_mode)
    for trans_path in file_list:
        generate(trans_path, output_folder, model_folder_path, dotted_line, data_mode, clean_mode, gt_mode, preview_mode, rename)

