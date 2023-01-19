import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import random


def flatten(v):
    """
    Flatten a list of lists/tuples
    """

    return [x for y in v for x in y]


def find_max_epoch(path):
    """
    Find maximum epoch/iteration in path, formatted ${n_iter}.pkl
    E.g. 100000.pkl

    Parameters:
    path (str): checkpoint path
    
    Returns:
    maximum iteration, -1 if there is no (valid) checkpoint
    """

    files = os.listdir(path)
    epoch = -1
    for f in files:
        if len(f) <= 4:
            continue
        if f[-4:] == '.pkl':
            try:
                epoch = max(epoch, int(f[:-4]))
            except:
                continue
    return epoch


def print_size(net):
    """
    Print the number of parameters of a network
    """

    if net is not None and isinstance(net, keras.layers.Layer):
        # module_parameters = net.all_params
        params = net.build([14,256]).count_params()
            # sum([np.prod(tf.shape(p)) for p in module_parameters])
        print("{} Parameters: {:.6f}M".format(
            net.__class__.__name__, params / 1e6), flush=True)


# Utilities for diffusion models

def std_normal(size):
    """
    Generate the standard Gaussian variable of a certain size
    """

    return tf.random.normal(size)


def calc_diffusion_step_embedding(diffusion_steps, diffusion_step_embed_dim_in):
    """
    Embed a diffusion step $t$ into a higher dimensional space
    E.g. the embedding vector in the 128-dimensional space is
    [sin(t * 10^(0*4/63)), ... , sin(t * 10^(63*4/63)), cos(t * 10^(0*4/63)), ... , cos(t * 10^(63*4/63))]

    Parameters:
    diffusion_steps (torch.long tensor, shape=(batchsize, 1)):     
                                diffusion steps for batch data
    diffusion_step_embed_dim_in (int, default=128):  
                                dimensionality of the embedding space for discrete diffusion steps
    
    Returns:
    the embedding vectors (torch.tensor, shape=(batchsize, diffusion_step_embed_dim_in)):
    """

    assert diffusion_step_embed_dim_in % 2 == 0

    half_dim = diffusion_step_embed_dim_in // 2
    _embed = np.log(10000) / (half_dim - 1)
    _embed = tf.convert_to_tensor(np.exp(np.arange(half_dim) * -_embed))
    _embed = tf.cast(diffusion_steps,_embed.dtype) * _embed
    diffusion_step_embed = tf.concat([tf.math.sin(_embed),
                                      tf.math.cos(_embed)], 1)

    return diffusion_step_embed


def calc_diffusion_hyperparams(T, beta_0, beta_T):
    """
    Compute diffusion process hyperparameters

    Parameters:
    T (int):                    number of diffusion steps
    beta_0 and beta_T (float):  beta schedule start/end value, 
                                where any beta_t in the middle is linearly interpolated
    
    Returns:
    a dictionary of diffusion hyperparameters including:
        T (int), Beta/Alpha/Alpha_bar/Sigma (torch.tensor on cpu, shape=(T, ))
        These cpu tensors are changed to cuda tensors on each individual gpu
    """

    Beta = np.linspace(beta_0, beta_T, T)  # Linear schedule
    Alpha = 1 - Beta
    Alpha_bar = Alpha + 0
    Beta_tilde = Beta + 0
    for t in range(1, T):
        Alpha_bar[t] *= Alpha_bar[t - 1]  # \bar{\alpha}_t = \prod_{s=1}^t \alpha_s
        Beta_tilde[t] *= (1 - Alpha_bar[t - 1]) / (
                1 - Alpha_bar[t])  # \tilde{\beta}_t = \beta_t * (1-\bar{\alpha}_{t-1})
        # / (1-\bar{\alpha}_t)
    Beta=tf.convert_to_tensor(Beta, dtype=tf.float32)
    Alpha = tf.convert_to_tensor(Alpha, dtype=tf.float32)
    Alpha_bar = tf.convert_to_tensor(Alpha_bar, dtype=tf.float32)
    Beta_tilde = tf.convert_to_tensor(Beta_tilde, dtype=tf.float32)

    Sigma = np.sqrt(Beta_tilde)  # \sigma_t^2  = \tilde{\beta}_t
    Sigma = tf.convert_to_tensor(Sigma, dtype=tf.float32)

    _dh = {}
    _dh["T"], _dh["Beta"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Sigma"] = T, Beta, Alpha, Alpha_bar, Sigma
    diffusion_hyperparams = _dh
    return diffusion_hyperparams


def sampling(net, size, diffusion_hyperparams, cond, mask, only_generate_missing=0, guidance_weight=0):
    """
    Perform the complete sampling step according to p(x_0|x_T) = \prod_{t=1}^T p_{\theta}(x_{t-1}|x_t)

    Parameters:
    net (torch network):            the wavenet model
    size (tuple):                   size of tensor to be generated, 
                                    usually is (number of audios to generate, channels=1, length of audio)
    diffusion_hyperparams (dict):   dictionary of diffusion hyperparameters returned by calc_diffusion_hyperparams
                                    note, the tensors need to be cuda tensors 
    
    Returns:
    the generated audio(s) in torch.tensor, shape=size
    """

    _dh = diffusion_hyperparams
    T, Alpha, Alpha_bar, Sigma = _dh["T"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Sigma"]
    assert len(Alpha) == T
    assert len(Alpha_bar) == T
    assert len(Sigma) == T
    assert len(size) == 3

    print('begin sampling, total number of reverse steps = %s' % T)

    x = std_normal(size)

    for t in range(T - 1, -1, -1):
        if only_generate_missing == 1:
            x = x * tf.cast(1 - mask,tf.float32) + cond *  tf.cast(mask,tf.float32)
        diffusion_steps = (t * tf.ones((size[0], 1)))  # use the corresponding reverse step
        epsilon_theta = net((x, cond, mask, diffusion_steps,))  # predict \epsilon according to \epsilon_\theta
        # update x_{t-1} to \mu_\theta(x_t)
        x = (x - (1 - Alpha[t]) / tf.math.sqrt(1 - Alpha_bar[t]) * epsilon_theta) / tf.math.sqrt(Alpha[t])
        if t > 0:
            x = x + Sigma[t] * std_normal(size)  # add the variance term to x_{t-1}

    return x


def training_loss(net, loss_fn, X, diffusion_hyperparams, only_generate_missing=1):
    """
    Compute the training loss of epsilon and epsilon_theta

    Parameters:
    net (torch network):            the wavenet model
    loss_fn (torch loss function):  the loss function, default is nn.MSELoss()
    X (torch.tensor):               training data, shape=(batchsize, 1, length of audio)
    diffusion_hyperparams (dict):   dictionary of diffusion hyperparameters returned by calc_diffusion_hyperparams
                                    note, the tensors need to be cuda tensors       
    
    Returns:
    training loss
    """

    _dh = diffusion_hyperparams
    T, Alpha_bar = _dh["T"], _dh["Alpha_bar"]

    finance = tf.cast(X[0],tf.float32)
    cond = X[1]
    gt_mask = X[2]
    loss_mask = X[3]
    observed_mask = X[4]

    B, C, L = finance.shape  # B is batchsize, C=1, L is audio length
    diffusion_steps = tf.random.uniform(shape=(B,1,1), minval=0, maxval=T, dtype=tf.int32) # randomly sample diffusion steps from 1~T

    z = std_normal(finance.shape)
    if only_generate_missing == 1:
        z = finance * tf.cast(gt_mask, dtype = tf.float32) + z * tf.cast((1-gt_mask), dtype = tf.float32)
    transformed_X = tf.math.sqrt(tf.reshape(tf.gather_nd(Alpha_bar,diffusion_steps),diffusion_steps.shape)) * finance + tf.math.sqrt(
        1 - tf.reshape(tf.gather_nd(Alpha_bar,diffusion_steps),diffusion_steps.shape)) * z  # compute x_t from q(x_t|x_0)
    epsilon_theta = net(
        (transformed_X, cond, gt_mask, tf.reshape(diffusion_steps,[B, 1]),observed_mask))  # predict \epsilon according to \epsilon_\theta

    if only_generate_missing == 1:
        return loss_fn(epsilon_theta[loss_mask], z[loss_mask])
    elif only_generate_missing == 0:
        return loss_fn(epsilon_theta, z)


def get_mask(sample):
    """Get mask of random points (missing at random) across channels based on k,
    where k == number of data points. Mask of sample's shape where 0's to be imputed, and 1's to preserved
    as per ts imputers"""
    num_feature = 6
    mask = np.ones(sample.shape)
    length_index = np.arange(tf.shape(mask)[0])  # lenght of series indexes
    np.random.shuffle(length_index)
    idxDJ = length_index[0:7] # DJ
    mask[:,:3][idxDJ] = 0
    np.random.shuffle(length_index)
    idxEU = length_index[0:2] # EU
    mask[:, 3:6][idxEU] = 0
    np.random.shuffle(length_index)
    idxHK = length_index[0:12] # HK
    mask[:, 6:][idxHK] = 0
    for channel in range(3):
        np.random.shuffle(length_index)
        idx = length_index[0: 8]
        for i in range(6//num_feature):
            mask[:, 3 + channel][idx] = 0
    return tf.convert_to_tensor(mask)



def get_mask_rm(sample, k):
    """Get mask of random points (missing at random) across channels based on k,
    where k == number of data points. Mask of sample's shape where 0's to be imputed, and 1's to preserved
    as per ts imputers"""

    mask = np.ones(sample.shape)
    length_index = np.arange(tf.shape(mask)[0])  # lenght of series indexes
    for channel in range(tf.shape(mask)[1]):
        np.random.shuffle(length_index)
        idx = length_index[0: k]
        mask[:, channel][idx] = 0

    return tf.convert_to_tensor(mask)


def get_mask_mnr(sample, k):
    """Get mask of random segments (non-missing at random) across channels based on k,
    where k == number of segments. Mask of sample's shape where 0's to be imputed, and 1's to preserved
    as per ts imputers"""

    mask = np.ones(sample.shape)
    length_index = np.arange(mask.shape[0])
    list_of_segments_index = np.split(length_index, k)
    for channel in range(mask.shape[1]):
        s_nan = random.choice(list_of_segments_index)
        mask[:, channel][s_nan[0]:s_nan[-1] + 1] = 0

    return tf.convert_to_tensor(mask)


def get_mask_bm(sample, k):
    """Get mask of same segments (black-out missing) across channels based on k,
    where k == number of segments. Mask of sample's shape where 0's to be imputed, and 1's to be preserved
    as per ts imputers"""

    mask = np.ones(sample.shape)
    length_index = np.arange(mask.shape[0])
    list_of_segments_index = np.split(length_index, k)
    s_nan = random.choice(list_of_segments_index)
    for channel in range(mask.shape[1]):
        mask[:, channel][s_nan[0]:s_nan[-1] + 1] = 0

    return tf.convert_to_tensor(mask)

def mask_missing_train_bm(data, k):
    observed_values = np.array(data)
    observed_masks = ~np.isnan(observed_values)
    gt_masks = observed_masks.copy()
    gt_masks[:,75:][-1] = 0

    observed_values = np.nan_to_num(observed_values)
    observed_masks = observed_masks.astype("float32")
    gt_masks = gt_masks.astype("float32")

    return observed_values, observed_masks, gt_masks


def time_embedding(pos, d_model=128):
    pe = np.zeros([pos.shape[0], pos.shape[1], d_model])
    position = tf.expand_dims(pos, 2)
    div_term = 1 / tf.math.pow(10000.0, tf.range(0, d_model, 2) / d_model)
    pe[:, :, 0::2] = np.sin(position.numpy() * div_term.numpy())
    pe[:, :, 1::2] = np.cos(position.numpy() * div_term.numpy())
    return tf.convert_to_tensor(pe)