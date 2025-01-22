import sys
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)

from render_lib import *

# Example usage
json_folder_path = "/Users/chenyangxu/Library/Mobile Documents/com~apple~CloudDocs/School/NYU 2024 Winter/AI4CE/models/breaking-bad preprocessing/trans_samples_large_constrained/3515.json"
model_folder_path = "/Users/chenyangxu/Library/Mobile Documents/com~apple~CloudDocs/School/NYU 2024 Winter/AI4CE/models/breaking-bad preprocessing"
output_folder = "output"
single_mode = True
file_num = 10
dotted_line = False

# json_folder_path = "your transformation json file/your folder with transformation json files"
# model_folder_path = "your breaking bad folder"
# output_folder = "your output folder"
# single_mode = True
# file_num = 100 
# dotted_line = True


if __name__ == "__main__":
    file_list = select_model(json_folder_path, file_num, single_mode)
    for json_path in file_list:
        generate(json_path, output_folder, model_folder_path, dotted_line)

