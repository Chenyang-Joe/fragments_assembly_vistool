import subprocess
import glob
import os
import sys
import random
from math import ceil

def create_video_mosaic(input_folder, rows, columns, output_file='output_mosaic.mkv', codec='h264_nvenc'):
    """
    Creates a video mosaic with the specified number of rows and columns from input videos.

    Args:
        input_folder (str): Path to the folder containing input videos.
        rows (int): Number of rows in the mosaic.
        columns (int): Number of columns in the mosaic.
        output_file (str): Name of the output mosaic video file.
        codec (str): Video codec to use for encoding (default: 'h264_nvenc').
    """
    # Calculate the total number of inputs needed
    total_inputs = rows * columns

    # Define the input file pattern
    # input_pattern = os.path.join(input_folder, 'resized_*.mp4')
    input_pattern = os.path.join(input_folder, '[0-9][0-9][0-9][0-9].mp4')
    
    # Retrieve and sort the input files
    input_files = sorted(glob.glob(input_pattern))
    random.shuffle(input_files)
    
    # Check if there are enough input files
    if len(input_files) < total_inputs:
        print(f"Error: Found {len(input_files)} input files, but {total_inputs} are required for a {rows}x{columns} mosaic.")
        sys.exit(1)
    
    # Select the required number of input files for the mosaic
    selected_inputs = input_files[:total_inputs]
    print(f"Selected {total_inputs} input files for a {rows}x{columns} mosaic:")
    for idx, file in enumerate(selected_inputs, 1):
        print(f"  Input {idx}: {file}")
    
    # Start building the ffmpeg command
    ffmpeg_cmd = ['ffmpeg']
    
    # Add each input file to the command
    for input_file in selected_inputs:
        ffmpeg_cmd.extend(['-i', input_file])
    
    # Construct the filter_complex string
    filter_complex_parts = []
    
    # Apply setpts to synchronize the videos and label them as a0, a1, ..., aN
    for idx in range(total_inputs):
        # filter_complex_parts.append(f'[{idx}:v] setpts=PTS-STARTPTS, scale=160x90 [a{idx}]')
        filter_complex_parts.append(f'[{idx}:v] setpts=PTS-STARTPTS, scale=640x360 [a{idx}]')
    
    # Generate the layout string for xstack
    layout_positions = []
    for row in range(rows):
        for col in range(columns):
            if col == 0:
                x_pos = "0"
            else:
                x_pos = "h0"
                for i in range(col-1):
                    x_pos += f"+h{i+1}"
            if row == 0:
                y_pos = "0"
            else:
                y_pos = "w0"
                for i in range(row-1):
                    y_pos += f"+w{i+1}"
            layout_positions.append(f"{y_pos}_{x_pos}")
    
    # Join the layout positions with '|'
    layout_str = '|'.join(layout_positions)
    print(f"\nConstructed layout string: {layout_str}")
    
    # Prepare the xstack filter inputs
    stack_inputs = ''.join([f'[a{idx}]' for idx in range(total_inputs)])
    print(f"\nConstructed xstack inputs: {stack_inputs}")
    
    # Add the xstack filter
    filter_complex_parts.append(f"{stack_inputs}xstack=inputs={total_inputs}:layout={layout_str}")
    
    # Combine all parts into a single filter_complex string separated by semicolons
    filter_complex = '; '.join(filter_complex_parts)
    # filter_complex = f'"{filter_complex}"'
    
    # Debug: Print the filter_complex string
    print("\nConstructed filter_complex:")
    print(filter_complex)
    
    # Add the filter_complex to the ffmpeg command
    ffmpeg_cmd.extend(['-filter_complex', filter_complex])
    
    # Specify the video codec
    ffmpeg_cmd.extend(['-c:v', codec])

    ffmpeg_cmd.extend(['-qp', '18']) # was 23

    if codec == "hevc_nvenc":
        ffmpeg_cmd.extend(['-vtag', 'hvc1'])

    # Add the output file to the ffmpeg command
    ffmpeg_cmd.append(output_file)

    # Print the constructed ffmpeg command for verification
    print("\nConstructed ffmpeg command:")
    print(' '.join(ffmpeg_cmd))
    
    # Execute the ffmpeg command
    try:
        subprocess.run(ffmpeg_cmd, check=True)
        print(f"\nMosaic video created successfully: {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"\nAn error occurred while creating the mosaic: {e}")
        sys.exit(1)

if __name__ == "__main__":
    import argparse

    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Create a video mosaic with specified rows and columns.")
    parser.add_argument('--input_folder', type=str, default='mosaic/resized', help='Path to the folder containing input videos.')
    parser.add_argument('--rows', type=int, default=25, help='Number of rows in the mosaic.')
    parser.add_argument('--columns', type=int, default=25, help='Number of columns in the mosaic.')
    parser.add_argument('--output_file', type=str, default='output_mosaic.mp4', help='Name of the output mosaic video file.')
    parser.add_argument('--codec', type=str, default='h264_nvenc', help='Video codec to use for encoding (default: h264_nvenc).')

    args = parser.parse_args()

    # Create the video mosaic with the provided arguments
    create_video_mosaic(args.input_folder, args.rows, args.columns, args.output_file, args.codec)

# python video_mosaic2.py --input_folder output_log1/video_selected --rows 9 --columns 7 --codec libx264
