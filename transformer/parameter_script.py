from pathlib import Path

class MODE:
    TRAIN = "train"
	GENERATE = "generate"

class ID:
	HAO = 0
	TONY = 1
	TYLER = 2

class MOOD:
	BRIGHT = "bright"
	SAD = "sad"

class GENRE:
	POP = "pop"
	ROCK = "rock"
	JAZZ = "jazz"
	CLASSICAL = "classical"


def tony_command_generator(mode, mood, genre, num_outputs, output_dir):
	if genre == GENRE.POP:
		MODEL_PATH = '/mnt/raid5/Projects/viMusic/tony/transformer/data/cached/models/Pop_Transformer.pth'
	elif genre == GENRE.ROCK:
		MODEL_PATH = '/mnt/raid5/Projects/viMusic/tony/transformer/data/cached/models/Rock_Transformer.pth'

	SCRIPT_PATH = '/mnt/raid5/Projects/viMusic/tony/transformer/generate_mt.py'
	MODE = 'gfm'

	model_path = str(Path(MODEL_PATH))
	script_path = str(Path(SCRIPT_PATH))
	mode = MODE
	num_outputs = str(num_outputs)
    n_words = str(600)

    command = ["python", script_path, \
				"--model_path", run_dir, \
				"--output_dir", output_dir,\
				"--num_outputs", num_outputs, \
				"--n_words", num_steps]

	command_in_use = " ".join(command)
	return command_in_use
def hao_command_generator(mode, mood, genre, num_outputs, output_dir):
	###### Modifiable #######
	if mood == MOOD.BRIGHT:
		CHECKPOINT_DIR = "/mnt/raid5/Projects/viMusic/hao/dataset/Pop_Bright_Sad_Classified/bright/checkpoint"
	elif mood == MOOD.SAD:
		CHECKPOINT_DIR = "/mnt/raid5/Projects/viMusic/hao/dataset/Pop_Bright_Sad_Classified/sad/checkpoint"

	SCRIPT_PATH = '/mnt/raid5/Projects/viMusic/hao/repo/viMusic/magenta/models/polyphony_rnn/polyphony_rnn_generate.py'
	HPARAMS = "'batch_size=64,rnn_layer_sizes=[128,128,128]'"
	#########################

	run_dir = str(Path(CHECKPOINT_DIR))
	script_path = str(Path(SCRIPT_PATH))
	hparams = HPARAMS
	num_outputs = str(num_outputs)
	num_steps = str(128)

	command = ["python", script_path, \
				"--run_dir", run_dir, \
				"--hparams",	hparams, \
				"--output_dir", output_dir,\
				"--num_outputs", num_outputs, \
				"--num_steps", num_steps]

	command_in_use = " ".join(command)
	return command_in_use

def tyler_command_generator(mode, mood, genre, num_outputs, output_dir):
	pass

def interface(mode, mood, genre, num_outputs, person_id):
	USING_GPU_NO = 1

	_mood = mood
	_genre = genre
	_task_name = mode
	_person_id = person_id

	output_dir = "../sample/temp_midi/"
	command = ''
	if _task_name == MODE.GENERATE:
		if _person_id == ID.TONY:
			command = tony_command_generator(mode, mood, genre, num_outputs, output_dir)

		elif _person_id == ID.HAO:
			command = hao_command_generator(mode, mood, genre, num_outputs, output_dir)

		elif _person_id == ID.TYLER:
			command = tyler_command_generator(mode, mood, genre, num_outputs, output_dir)

	return command

# test function
mode = "generate"
mood = MOOD.SAD
genre = GENRE.POP
num_outputs = 1
person_id = ID.HAO
print (interface(mode, mood, genre, num_outputs, person_id))

