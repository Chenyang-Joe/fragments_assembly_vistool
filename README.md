
```markdown
# README

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

---

## Set Up Parameters

Open and edit the parameters in **`render_script.py`** as needed:

### 1. Render Multiple Models

- `json_folder_path` = `"path/to/transformation/json/files"`
- `model_folder_path` = `"path/to/model/folder"`
- `output_folder`     = `"path/to/output/folder"`
- `single_mode`       = `False`
- `file_num`          = `100`  
  *(Number of files randomly drawn from the JSON folder.)*

### 2. Render a Single Model

- `json_folder_path` = `"path/to/transformation/json/file"`
- `model_folder_path` = `"path/to/model/folder"`
- `output_folder`     = `"path/to/output/folder"`
- `single_mode`       = `True`
- `file_num`          = `100`  
  *(You can set this to any number you prefer.)*

---

## JSON File Format

Each JSON file should include:
- **Model path**
- **Ground truth transformation**
- **Prediction transformation**

For examples, see the JSON files under the `transformation_data_example` directory.
```

Feel free to adjust paths or numbers according to your specific setup.# fragments_assembly_vistool
