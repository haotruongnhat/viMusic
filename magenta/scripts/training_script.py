from pathlib import Path
import pyperclip

# Constanst
MODELS_FOLDER = "magenta/models"
CREATE_DATASET = "_create_dataset"
TRAIN = "_train"
GENERATE = "_generate"
####################
#### Modifiable ####
MODEL_NAME = "polyphony_rnn"
TASK_NAME = CREATE_DATASET

CHECKPOINT_DIR = ""
HPARAMS = "batch_size=128,rnn_layer_sizes=[128,128,128]"

####################
model_folder = Path(MODELS_FOLDER) / MODEL_NAME
model_name = MODEL_NAME
task_name = TASK_NAME

script_path = model_folder / (MODEL_NAME + TASK_NAME + ".py")
script_path = str(script_path)
if TASK_NAME == CREATE_DATASET:
	#### Modifiable ####
	INPUT_DIR = "/Users/hao/Desktop/Projects/viMusic/dataset/bright"
	OUTPUT_DIR = "/Users/hao/Desktop/Projects/viMusic/dataset/bright"
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
	SEQUENCE_EXAMPLE_FILE = ""
	NUM_TRAINING_STEPS = 20000
	####################
	hparams = HPARAMS
	run_dir = Path(CHECKPOINT_DIR)
	sequence_example_file = Path(SEQUENCE_EXAMPLE_FILE)

	run_dir = str(run_dir)
	sequence_example_file = str(sequence_example_file)

	command = ["python", script_path, \
					"--run_dir", run_dir, \
					"--sequence_example_file",	sequence_example_file, \
					"--hparams", hparams,\
					"--num_training_steps", NUM_TRAINING_STEPS]
elif task_name == GENERATE:
	#### Modifiable ####
	OUTPUT_DIR = ""
	####################
	hparams = HPARAMS
	run_dir = Path(CHECKPOINT_DIR)
	output_dir = Path(OUTPUT_DIR)

	run_dir = str(run_dir)
	output_dir = str(output_dir)

	command = ["python", script_path, \
					"--run_dir", run_dir, \
					"--hparams",	hparams, \
					"--output_dir", output_dir,\
					"--num_outputs", 10, \
					"--num_steps", 128]

command_in_use = " ".join(command)
pyperclip.copy(command_in_use)
print(command_in_use)
		





