import sys
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)

from render_lib import *

# Example usage
json_folder_path = "/Users/chenyangxu/Library/Mobile Documents/com~apple~CloudDocs/School/NYU 2024 Winter/AI4CE/models/breaking-bad preprocessing/flow_mathcing trans/flow_matching_everyday_vol"
model_folder_path = "/Users/chenyangxu/Library/Mobile Documents/com~apple~CloudDocs/School/NYU 2024 Winter/AI4CE/models/breaking-bad preprocessing"
output_folder = "output"
single_mode = False
all_files = True
random_draw = True
range = [0,10]
file_num = 10
dotted_line = False


if __name__ == "__main__":
    file_list = select_model(json_folder_path, file_num, single_mode, random_draw, all_files, range)
    for json_path in file_list:
        generate(json_path, output_folder, model_folder_path, dotted_line)

