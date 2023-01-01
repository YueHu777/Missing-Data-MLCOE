import os
import argparse
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras

from utils.util import get_mask_mnr, get_mask_bm, get_mask_rm
from utils.util import find_max_epoch, print_size, sampling, calc_diffusion_hyperparams

from imputers.CSDI import CSDIImputer
from sklearn.metrics import mean_squared_error
from statistics import mean


def generate(output_directory,
             num_samples,
             ckpt_path,
             data_path,
             ckpt_iter,
             use_model,
             masking,
             missing_k,
             only_generate_missing):
    """
    Generate data based on ground truth

    Parameters:
    output_directory (str):           save generated speeches to this path
    num_samples (int):                number of samples to generate, default is 4
    ckpt_path (str):                  checkpoint path
    ckpt_iter (int or 'max'):         the pretrained checkpoint to be loaded;
                                      automitically selects the maximum iteration if 'max' is selected
    data_path (str):                  path to dataset, numpy array.
    use_model (int):                  0:DiffWave. 1:SSSDSA. 2:SSSDS4.
    masking (str):                    'mnr': missing not at random, 'bm': black-out, 'rm': random missing
    only_generate_missing (int):      0:all sample diffusion.  1:only apply diffusion to missing portions of the signal
    missing_k (int)                   k missing time points for each channel across the length.
    """

    model = CSDIImputer()
    # Get shared output_directory ready
    epoch_no = 120
    checkpoint_name = '{}CSDI.pkl'.format(epoch_no)
    model_path = os.path.join(output_directory, checkpoint_name)
    model.load_weights(model_path)
    print('Successfully loaded model at iteration {}'.format(epoch_no))
    masking = 'rm'
    missing_ratio = 0.05


    ### Custom data loading and reshaping ###

    testing_data = np.load(trainset_config['test_data_path'])
    testing_data = tf.convert_to_tensor(testing_data, tf.float32)
    print('Data loaded')

    imputations, targets = model.impute(testing_data, masking, 4)
    print(imputations[0].shape)
    # batch = batch.numpy()
    outfile = f'CSDIimputation.npy'
    new_out = os.path.join(output_directory, outfile)
    np.save(new_out, imputations)

    outfile = f'CSDItargets.npy'
    new_out = os.path.join(output_directory, outfile)
    np.save(new_out, targets)
    print('Data Saved')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/config_SSSDS4.json',
                        help='JSON file for configuration')
    parser.add_argument('-ckpt_iter', '--ckpt_iter', default='max',
                        help='Which checkpoint to use; assign a number or "max"')
    parser.add_argument('-n', '--num_samples', type=int, default=500,
                        help='Number of utterances to be generated')
    args = parser.parse_args()


    # Parse configs. Globals nicer in this case
    with open(args.config) as f:
        data = f.read()
    config = json.loads(data)
    print(config)

    gen_config = config['gen_config']

    train_config = config["train_config"]  # training parameters

    global trainset_config
    trainset_config = config["trainset_config"]  # to load trainset

    global diffusion_config
    diffusion_config = config["diffusion_config"]  # basic hyperparameters

    global diffusion_hyperparams
    diffusion_hyperparams = calc_diffusion_hyperparams(
        **diffusion_config)  # dictionary of all diffusion hyperparameters

    global model_config
    if train_config['use_model'] == 0:
        model_config = config['wavenet_config']
    elif train_config['use_model'] == 1:
        model_config = config['sashimi_config']
    elif train_config['use_model'] == 2:
        model_config = config['wavenet_config']
    generate(**gen_config,
             ckpt_iter=args.ckpt_iter,
             num_samples=args.num_samples,
             use_model=train_config["use_model"],
             data_path=trainset_config["test_data_path"],
             masking=train_config["masking"],
             missing_k=train_config["missing_k"],
             only_generate_missing=train_config["only_generate_missing"])
