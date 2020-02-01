from pathlib import Path
import pyperclip

### Sequence related
#########
SCRIPT_DIR = "magenta/scripts"

INPUT_DIR = "/Users/hao/Desktop/Projects/viMusic/dataset/sad"
OUTPUT_DIR = "/Users/hao/Desktop/Projects/viMusic/dataset/sad"

RECURSIVE = False
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

command_in_use = " ".join(command)
pyperclip.copy(command_in_use)
print(command_in_use)
