import sys
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)

from render_lib import *

# Example usage
trans_folder_path = "/Users/chenyangxu/Library/Mobile Documents/com~apple~CloudDocs/School/NYU 2024 Winter/AI4CE/models/assembly_recording/puzzle_fussion_trans"
model_folder_path = "/Users/chenyangxu/Library/Mobile Documents/com~apple~CloudDocs/School/NYU 2024 Winter/AI4CE/models/breaking-bad preprocessing"
output_folder = "output"
single_mode = False
traversal_all = True
random_draw = False
range = [0,10]
file_num = 10
dotted_line = False
data_mode = "puzzlefusion" #"jigsaw", "puzzlefusion", "flowmatching"
clean_mode = True


if __name__ == "__main__":
    file_list = select_model(trans_folder_path, file_num, single_mode, random_draw, traversal_all, range, data_mode)
    for trans_path in file_list:
        generate(trans_path, output_folder, model_folder_path, dotted_line, data_mode, clean_mode)

