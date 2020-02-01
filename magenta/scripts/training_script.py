import os, sys
sys.path.append(os.getcwd())
import time
from pathlib import Path

# Constanst
MODEL_FOLDER = "magenta/models"
CREATE_DATASET = "_create_dataset"
TRAIN = "_train"
GENERATE = "_generate"
####################
#### Modifiable ####
MODEL_NAME = "polyphony_rnn"
TASK_NAME = CREATE_DATASET

####################
model_folder = Path(MODEL_FOLDER)
model_name = MODEL_NAME
task_name = TASK_NAME

script_path = model_folder / (MODEL_NAME + TASK_NAME + ".py")


if TASK_NAME == CREATE_DATASET:
	#### Modifiable ####
	INPUT_DIR = ""

	OUTPUT_DIR = ""
	####################
	input_path = Path(INPUT_DIR) / "notesequences.tfrecord"
	output_dir = Path(OUTPUT_DIR)

	input_path = str(input_path)
	output_dir = str(output_dir)

	command = ["python", script_path, \
							"--input", input_path, \
							"--output_dir",	output_dir, \
							"--eval_ratio", "0.10"]
elif TASK_NAME == TRAIN:
  #### Modifiable ####
	RUN_DIR = ""
	SEQUENCE_EXAMPLE_FILE = ""
	NUM_TRAINING_STEPS = 20000
	HPARAMS = "batch_size=64,rnn_layer_sizes=[64,64]"
	####################
	hparams = HPARAMS
	run_dir = Path(RUN_DIR)
	sequence_example_file = Path(SEQUENCE_EXAMPLE_FILE)

	run_dir = str(run_dir)
	sequence_example_file = str(sequence_example_file)

	command = ["python", script_path, \
					"--run_dir", run_dir, \
					"--sequence_example_file",	sequence_example_file, \
					"--hparams", hparams,\
					"--num_training_steps", NUM_TRAINING_STEPS]

print(command)
		





