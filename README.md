## Environment

1. **Install Blender**  
   - Recommended version: **4.3.2** (the newest at the time of writing)

2. **Create a Conda environment**  
   ```bash
   conda create -n blender python=3.10
   ```

3. **Activate the Conda environment**  
   ```bash
   conda activate blender
   ```

4. **Install `bpy`**  
   ```bash
   pip install bpy
   ```

5. **Install `ffmpeg`**  
   ```bash
   conda install -c conda-forge ffmpeg
   ```

---

## Set Up Parameters

Open and edit the parameters in **`render_script.py`** as needed:

- `json_folder_path` = `"path/to/transformation/folder"`
- `model_folder_path` = `"path/to/model/folder"`
- `output_folder`     = `"path/to/output/folder"`
- `single_mode`       = `False`
- `dotted_line`       = `True`  
- `random_draw`       = `True`
- `all_files`          = `True`
- `range`             = `[0,7871]`
- `file_num`          = `100`  
- `data_mode`         = `mode name` 
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