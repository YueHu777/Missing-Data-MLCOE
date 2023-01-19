from imputers.CSDI import CSDIImputer
import os
import argparse
import json
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/config_SSSDS4.json',
                        help='JSON file for configuration')

    args = parser.parse_args()

    with open(args.config) as f:
        data = f.read()

    config = json.loads(data)
    print(config)

    train_config = config["train_config"]  # training parameters

    global trainset_config
    trainset_config = config["trainset_config"]  # to load trainset

    training_data = np.load(trainset_config['train_data_path'])
    training_data = tf.convert_to_tensor(training_data,tf.float32)
    print('Data loaded')

    imputer = CSDIImputer()
    masking = 'rm'
    missing_ratio = 0.05
    path_save = config["train_config"]['output_directory']

    imputer.train(training_data, masking, missing_ratio, path_save)
