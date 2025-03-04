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

5. **Install `trimesh` (optional for removal)**
   `trimesh` must be installed in python env concluded in blender. Find blender python env first
   ```bash
   blender --background --python-expr "import sys; print(sys.executable)"   
   ```
   install trimesh 
   ```
   /python/path/example/python/bin/python3.11 -m ensurepip  
   /python/path/example/python/bin/python3.11 -m pip install trimesh
   blender --background --python-expr "import trimesh; print(trimesh.__version__)"      
   ```
---

## Set Up Parameters

Open and edit the parameters in **`config.json`** as needed:

   ```bash
   {
      "trans_folder_path": "PATH",
      "model_folder_path": "PATH",
      "output_folder": "PATH",
      "single_mode": false,
      "traversal_all": false,
      "random_draw": false,
      "range": [0, 10],
      "file_num": 10,
      "dotted_line": false,
      "data_mode": "flowmatching",
      "clean_mode": true,
      "gt_mode": false,
      "preview_mode": true,
      "preview_rotate": true,
      "rename": true
   }
   ```


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