## Environment

1. **Install Blender**  
   - Recommended version: **4.3.2** (the newest at the time of writing)

2. **Create a Conda environment**  
   ```bash
   conda create -n blender python=3.11
   ```

3. **Activate the Conda environment**  
   ```bash
   conda activate blender
   ```

4. **Install `ffmpeg`**  
   ```bash
   conda install -c conda-forge ffmpeg
   ```

5. **Install `hp5y` (optional for removal)**
   `hp5y` must be installed in python env concluded in blender. Find blender python env first
   ```bash
   blender --background --python-expr "import sys; print(sys.executable)"   
   ```
   install hp5y 
   ```
   /python/path/example/python/bin/python3.11 -m ensurepip  
   /python/path/example/python/bin/python3.11 -m pip install h5py
   blender --background --python-expr "import h5py; print(h5py.__version__)"      
   ```
---

## Set Up Parameters

Open and edit the parameters in **`render_script.py`** as needed:

- `json_folder_path` = `"path/to/transformation/folder"`
- `model_folder_path` = `"path/to/model/folder"`
- `output_folder`     = `"path/to/output/folder"`
- `removal_meta_file` = `"path/to/removal/file"`
- `removal_num` = `4`
- `removal_mode` = `True`
- `single_mode`       = `False # render a given file or multiple files in given folder`
- `dotted_line`       = `True`  
- `random_draw`       = `True`
- `all_files`          = `True # traverse all files or files of the given range and given number`
- `range`             = `[0,7871]`
- `file_num`          = `100`  
- `data_mode`         = `mode_name #"jigsaw", "puzzlefusion", "flowmatching"`
- `clean_mode`        = `True`


---

## JSON File Format

Each JSON file should include:
- **Model path**
- **Ground truth transformation**
- **Prediction transformation**

For examples, see the JSON files under the `transformation_data_example` directory.

---

## Run Script
```bash
blender --background --python render_script.py          
```
---

## video_mosaic2

This is a tool to create mosaic video. If you only want to visualise the assembly result, it is not required.