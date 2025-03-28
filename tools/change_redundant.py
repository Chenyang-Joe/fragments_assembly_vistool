import os
import json
import h5py

folder = "/mnt/NAS/data/assembly/last_week/dataset/breakingbad_fantastic/GARF/artifact_redundant_4/version_0/epoch-0/json_results"
folder_list = os.listdir(folder)
h_file = "/mnt/NAS/data/assembly/hdf5/breaking_bad_vol.hdf5"

count = 0
with h5py.File(h_file, 'r') as hdf:
    for json_file_name in folder_list:
        print("step:", count)
        count += 1
        json_file_path = os.path.join(folder, json_file_name)
        

        with open(json_file_path, 'r+') as json_file:
            data = json.load(json_file)
            
            name = data.get('name', "")

            rdd_str = data.get('redundant_pieces', "")
            new_str = ''

            rdd_list = rdd_str.split(',')
            for rdd_name in rdd_list:
                rdd_parts = '/'.join(rdd_name.split('/')[:-2])
                rdd_idx = int(rdd_name.split('/')[-1])
                pieces_names = hdf[f"{rdd_parts}/pieces_names"][:]
                new_piece = str(pieces_names[rdd_idx])[2:-1]
                new_rdd_name = rdd_parts+'/'+new_piece
                if new_str == '':
                    new_str = new_rdd_name
                else:
                    new_str += ',' + new_rdd_name

            data['redundant_pieces'] = new_str
            
            json_file.seek(0)
            
            json.dump(data, json_file, indent=4)
            
            json_file.truncate()

print("All JSON files have been processed.")
