import glob
import multiprocessing as mp
import os
import time
from argparse import ArgumentParser
from pathlib import Path
import tensorflow as tf
import numpy as np
from tqdm import tqdm

from data.preprocess import generate_flying_things_point_cloud, get_all_flying_things_frames
from data.preprocess import preprocess, merge_metadata

global output_directory


def preprocess_wrap(tfrecord_file):
    preprocess(tfrecord_file, output_directory, frames_per_segment=None)


# https://github.com/tqdm/tqdm/issues/484
if __name__ == '__main__':
    tf.config.set_visible_devices([], 'GPU')
    visible_devices = tf.config.get_visible_devices()
    for device in visible_devices:
        assert device.device_type != 'GPU'
    parser = ArgumentParser()
    parser.add_argument('input_directory', type=str)
    parser.add_argument('output_directory', type=str)
    parser.add_argument('--n_cores', default=None, type=int)
    args = parser.parse_args()

    print(f"Extracting frames from {args.input_directory} to {args.output_directory}")

    input_directory = Path(args.input_directory)
    if not input_directory.exists() or not input_directory.is_dir():
        print("Input directory does not exist")
        exit(1)

    output_directory = Path(args.output_directory)
    if not output_directory.exists():
        output_directory.mkdir(parents=True)
    #if list(output_directory.iterdir()):
    #    print("Output directory not empty! Please remove existing files as there is no merge.")
    #    exit(1)
    output_directory = os.path.abspath(output_directory)

    n_cores = mp.cpu_count()
    if args.n_cores is not None:
        if args.n_cores <= 0:
            print("Number of cores cannot be negative")
            exit(1)
        if args.n_cores > n_cores:
            print(f"Number of cores cannot be more than{n_cores}")
            exit(1)
        else:
            n_cores = args.n_cores

    print(f"{n_cores} number of cores available")

    tfrecord_filenames = []
    os.chdir(input_directory)
    for file in glob.glob("*.tfrecord"):
        file_name = os.path.abspath(file)
        tfrecord_filenames.append(file_name)

    t = time.time()

    pool = mp.Pool(min(n_cores, len(tfrecord_filenames)))

    for _ in tqdm(pool.imap_unordered(preprocess_wrap, tfrecord_filenames), total=len(tfrecord_filenames)):
        pass

    # Close Pool and let all the processes complete
    pool.close()
    pool.join()  # postpones the execution of next line of code until all processes in the queue are done.

    # Merge look up tables
    print("Merging individual metadata...")
    merge_metadata(os.path.abspath(output_directory))

    print(f"Preprocessing duration: {(time.time() - t):.2f} s")


