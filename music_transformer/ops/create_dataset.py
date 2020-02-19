import argparse, warnings
from os import path
from music_transformer.data_pipeline import convert_directory,gen_tf_dataset_from_tfrecord
from music_transformer.utils import default_config

import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument('-idir','--input_dir',
help="Input directory used in generating songs into TFRecord",type=str)
parser.add_argument('-iofile','--first_tfrecord',
help="Input TFRecord used in transformation by data pipeline",type=str)
parser.add_argument('-r','--is_recursive',
help="Whether to read directory with recursive mode",type=bool)
parser.add_argument('-ofile','--second_tfrecord',
help='Data after being transformed',type=str)

class FirstOutputTFRecordPathError(Exception):
    pass

def main():
    args = parser.parse_args()
    if args.input_dir is not None and path.isdir(args.input_dir):
        default_first_tfrecord = path.join(args.input_dir,"tmp.tfrecord")
        if args.first_tfrecord is not None and \
            args.first_tfrecord.endswith(".tfrecord"):
                convert_directory(args.input_dir, args.first_tfrecord, args.is_recursive)
                if args.second_tfrecord is not None and args.second_tfrecord.endswith(".tfrecord"):
                    second_tfrecord_subdir = args.second_tfrecord[:args.second_tfrecord.rfind("/")]
                    if not path.isdir(second_tfrecord_subdir):
                        tf.io.gfile.mkdir(second_tfrecord_subdir)
                    gen_tf_dataset_from_tfrecord(args.first_tfrecord,args.second_tfrecord)
        else:
            warnings.warn("Unknown file path to first tfrecord. \
            Using default path: {}".format(default_first_tfrecord))
            convert_directory(args.input_dir, default_first_tfrecord, args.is_recursive)
            if args.second_tfrecord is not None and args.second_tfrecord.endswith(".tfrecord"):
                second_tfrecord_subdir = args.second_tfrecord[:args.second_tfrecord.rfind("/")]
                if not path.isdir(second_tfrecord_subdir):
                    tf.io.gfile.mkdir(second_tfrecord_subdir)
                gen_tf_dataset_from_tfrecord(default_first_tfrecord,args.second_tfrecord)
    else:
        if args.first_tfrecord is None or \
            not path.isfile(args.first_tfrecord):
            raise FirstOutputTFRecordPathError("\
                Please provide correct file path to store data")
        else:
            second_tfrecord_subdir = args.second_tfrecord[:args.second_tfrecord.rfind("/")]
            if args.second_tfrecord is not None and args.second_tfrecord.endswith(".tfrecord"):
                if not path.isdir(second_tfrecord_subdir):
                    tf.io.gfile.mkdir(second_tfrecord_subdir)
                gen_tf_dataset_from_tfrecord(args.first_tfrecord,args.second_tfrecord)
            else:
                first_tfrecord_subdir = args.first_tfrecord[:args.first_tfrecord.rfind("/")]
                default_second_tfrecord = path.join(first_tfrecord_subdir,"tmp_2.tfrecord")
                warnings.warn("Unknown file path to second tfrecord. \
                Using default path: {}".format(default_second_tfrecord))
                gen_tf_dataset_from_tfrecord(args.first_tfrecord,args.second_tfrecord)


if __name__ == "__main__":
    main()