import sys
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)

from render_lib import *

# Example usage
json_folder_path = "/Users/chenyangxu/Library/Mobile Documents/com~apple~CloudDocs/School/NYU 2024 Winter/AI4CE/models/assembly_recording/diffusion_sample_old/diffusion_trans_samples_large_constrained/123.json"
model_folder_path = "/Users/chenyangxu/Library/Mobile Documents/com~apple~CloudDocs/School/NYU 2024 Winter/AI4CE/models/breaking-bad preprocessing"
output_folder = "output"
single_mode = False
traversal_all = True
random_draw = False
range = [0,10]
file_num = 10
dotted_line = False
data_mode = "flowmatching" #"jigsaw", "puzzlefussion", "flowmatching"
clean_mode = True


if __name__ == "__main__":
    file_list = select_model(json_folder_path, file_num, single_mode, random_draw, traversal_all, range)
    for json_path in file_list:
        generate(json_path, output_folder, model_folder_path, dotted_line, data_mode, clean_mode)

