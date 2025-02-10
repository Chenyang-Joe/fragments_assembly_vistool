import sys
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)

from render_lib import *

# Example usage
trans_folder_path = "/Users/chenyangxu/Codebase/AI4CE/data_analysis/three_compare/scaling_random"
model_folder_path = "/Users/chenyangxu/Codebase/AI4CE/assembly/breaking_bad_model"
output_folder = "output"
removal_meta_file = "/Users/chenyangxu/Codebase/AI4CE/assembly/assembly_recodring_final/PuzzleFusion++/breaking_bad_vol_with_removal_keep_largest.hdf5" 
removal_num = 4
removal_mode = False
single_mode = False
traversal_all = True
random_draw = False
range = [0,10]
file_num = 10
dotted_line = False
data_mode = "flowmatching" #"jigsaw", "puzzlefusion", "flowmatching"
clean_mode = True
gt_mode = False


if __name__ == "__main__":
    file_list = select_model(trans_folder_path, file_num, single_mode, random_draw, traversal_all, range, data_mode)
    removal_dict = read_removal_dict(removal_meta_file, removal_num, removal_mode, file_list, data_mode)
    for trans_path in file_list:
        generate(trans_path, output_folder, model_folder_path, dotted_line, data_mode, clean_mode, gt_mode, removal_dict)

