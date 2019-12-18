import sys
from performance_rnn.utils import create_dataset, gen_midi_dataset_by_model
from performance_rnn.train_eval import train_performance_rnn, eval_performance_rnn
from performance_rnn.generate import generate_melody

def main():
    command = sys.argv[1]
    dataset_name = sys.argv[2] if len(sys.argv) >= 3 else None
    config = bool(sys.argv[3]) if len(sys.argv) >= 4 else "performance_with_dynamics"
    gen_by_model = bool(sys.argv[4]) if len(sys.argv) == 5 else False
    if command == 'create-dataset' and dataset_name is not None:
        create_dataset(dataset_name)
        if gen_by_model:
            gen_midi_dataset_by_model(dataset_name,config=config)
    elif command == 'train-restart':
        train_performance_rnn(dataset_name,config,False,training_steps=20000)
    elif command == 'train':
        train_performance_rnn(dataset_name,config,training_steps=5000)
    elif command == 'eval':
        eval_performance_rnn(dataset_name,config)
    elif command == 'generate':
        generate_melody(dataset_name,config,second=30)
        

if __name__ == '__main__':
    main()