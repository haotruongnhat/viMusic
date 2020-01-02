import sys
from rl_tuner.utils import create_dataset, gen_midi_dataset_by_model
from rl_tuner.train_eval import train_note_rnn, train_rl_tuner
from rl_tuner.generate import generate
def main():
    command = sys.argv[1]
    dataset_name = sys.argv[2]
    #create dataset
    #gen_midi_dataset_by_model(dataset_name)
    #pre_trained melody rnn
    #train_note_rnn(dataset_name)
    train_rl_tuner(dataset_name)
    #generate(dataset_name)


if __name__ == '__main__':
    main()