import sys
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)

from render_lib import *

# Example usage
trans_folder_path = "/Users/chenyangxu/Library/Mobile Documents/com~apple~CloudDocs/School/NYU 2024 Winter/AI4CE/models/assembly_recording/flow_mathcing trans/flow_matching_everyday_vol/6065.json"
model_folder_path = "/Users/chenyangxu/Library/Mobile Documents/com~apple~CloudDocs/School/NYU 2024 Winter/AI4CE/models/breaking-bad preprocessing/volume"
output_folder = "output"
single_mode = True
traversal_all = True
random_draw = False
range = [0,10]
file_num = 10
dotted_line = False
data_mode = "flowmatching" #"jigsaw", "puzzlefusion", "flowmatching"
clean_mode = False
gt_mode = False


if __name__ == "__main__":
    file_list = select_model(trans_folder_path, file_num, single_mode, random_draw, traversal_all, range, data_mode)
    for trans_path in file_list:
        generate(trans_path, output_folder, model_folder_path, dotted_line, data_mode, clean_mode, gt_mode)

