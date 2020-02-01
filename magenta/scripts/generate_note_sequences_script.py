import os, sys
sys.path.append(os.getcwd())
import time
from pathlib import Path


### Sequence related
#########
SCRIPT_DIR = "magenta/scripts"

INPUT_DIR = ""
OUTPUT_DIR = ""

RECURSIVE = True
script_path = Path(SCRIPT_DIR) / "convert_dir_to_note_sequences.py"
input_dir = Path(INPUT_DIR)
output_file = Path(OUTPUT_DIR) / "notesequences.tfrecord"

script_path = str(script_path)
input_dir = str(input_dir)
output_file = str(output_file)

command = ["python", \
            script_path, \
            "--input_dir", \
            INPUT_DIR, \
            "--output_file", \
            output_file, \
            "--recursive" if RECURSIVE else ""]

print(" ".join(command))
