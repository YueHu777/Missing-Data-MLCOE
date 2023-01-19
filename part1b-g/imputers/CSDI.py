import numpy as np
import random
import pandas as pd
import pickle
import math
import argparse
import datetime
import json
import os
import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
from rasa.utils.tensorflow.transformer import  TransformerEncoder
''' Standalone CSDI imputer. The imputer class is located in the last part of the notebook, please see more documentation there'''



def train(model, config, train_loader, path_save=""):
    p1 = int(0.75 * config["epochs"])
    p2 = int(0.9 * config["epochs"])
    lr_scheduler = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
                            boundaries=[p1, p2], values=get_lr_values(config["lr"],[p1, p2],0.1), name=None)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_scheduler)

    values_batch = tf.convert_to_tensor(cut(train_loader.observed_values, 16))
    masks_batch = tf.convert_to_tensor(cut(train_loader.observed_masks, 16))
    gt_batch = tf.convert_to_tensor(cut(train_loader.gt_masks, 16))
    timepoints = tf.convert_to_tensor(np.tile(np.arange(values_batch.shape[2]),[16,1]))

    epoch_no = 0
    # epoch_no = 60
    # checkpoint_name = '{}CSDI.pkl'.format(epoch_no)
    # model_path = os.path.join(path_save, checkpoint_name)
    # model.load_weights(model_path)
    # print('Successfully loaded model at iteration {}'.format(epoch_no))

    best_valid_loss = 1e10
    while epoch_no < config["epochs"]+1:
        avg_loss = 0
        # model.train()
        # with tqdm(training_data, mininterval=5.0, maxinterval=5.0) as it:
        for batch_no in range(len(values_batch)):
            X = values_batch[batch_no], masks_batch[batch_no], gt_batch[batch_no], timepoints
            with tf.GradientTape() as tape:
            # optimizer.zero_grad()
                loss = model(X)
            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            avg_loss += loss.numpy()
            if batch_no % 100 == 0:
                print("epoch: {} \tavg_epoch_loss: {}".format(epoch_no+1, avg_loss / (batch_no+1)))

        if epoch_no > 0 and epoch_no % 10 == 0:
            checkpoint_name = '{}CSDI.pkl'.format(epoch_no)
            output_path = os.path.join(path_save, checkpoint_name)
            model.save_weights(output_path)
            print('model at epoch_no %s is saved' % epoch_no)
        epoch_no += 1

    
def quantile_loss(target, forecast, q: float, eval_points) -> float:
    return 2 * torch.sum(torch.abs((forecast - target) * eval_points * ((target <= forecast) * 1.0 - q)))


def calc_denominator(target, eval_points):
    return torch.sum(torch.abs(target * eval_points))


def calc_quantile_CRPS(target, forecast, eval_points, mean_scaler, scaler):
    target = target * scaler + mean_scaler
    forecast = forecast * scaler + mean_scaler

    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = calc_denominator(target, eval_points)
    CRPS = 0
    for i in range(len(quantiles)):
        q_pred = []
        for j in range(len(forecast)):
            q_pred.append(torch.quantile(forecast[j: j + 1], quantiles[i], dim=1))
        q_pred = torch.cat(q_pred, 0)
        q_loss = quantile_loss(target, q_pred, quantiles[i], eval_points)
        CRPS += q_loss / denom
        
    return CRPS.numpy() / len(quantiles)


def evaluate(model, test_loader, nsample=100, scaler=1, mean_scaler=0, path_save=""):
    # model.eval()
    mse_total = 0
    mae_total = 0
    evalpoints_total = 0

    all_target = []
    all_observed_point = []
    all_observed_time = []
    all_evalpoint = []
    all_generated_samples = []

    values_batch = tf.convert_to_tensor(cut(test_loader.observed_values, 16))
    masks_batch = tf.convert_to_tensor(cut(test_loader.observed_masks, 16))
    gt_batch = tf.convert_to_tensor(cut(test_loader.gt_masks, 16))
    timepoints = tf.convert_to_tensor(np.tile(np.arange(values_batch.shape[2]), [16, 1]))

    for batch_no in range(5):
        X = values_batch[batch_no], masks_batch[batch_no], gt_batch[batch_no], timepoints
        output = model.evaluate(X, nsample)

        samples, c_target, eval_points, observed_points, observed_time = output
        samples = tf.transpose(samples,perm=[0, 1, 3, 2])  # (B,nsample,L,K)
        c_target = tf.transpose(c_target,perm=[0, 2, 1])  # (B,L,K)
        eval_points = tf.transpose(eval_points,perm=[0, 2, 1])
        observed_points = tf.transpose(observed_points,perm=[0, 2, 1])

        samples_median = tf.convert_to_tensor(np.median(samples.numpy()))
        all_target.append(c_target)
        all_evalpoint.append(eval_points)
        all_observed_point.append(observed_points)
        all_observed_time.append(observed_time)
        all_generated_samples.append(samples)

        mse_current = (((samples_median - c_target) * eval_points) ** 2) * (scaler ** 2)
        mae_current = (tf.math.abs((samples_median - c_target) * eval_points)) * scaler

        mse_total += tf.reduce_sum(mse_current).numpy()
        mae_total += tf.reduce_sum(mae_current).numpy()
        evalpoints_total += tf.reduce_sum(eval_points).numpy()

        # with open(f"{path_save}generated_outputs_nsample"+str(nsample)+".pk","wb") as f:
        #     all_target = tf.concat(all_target, axis=0)
        #     all_evalpoint = tf.concat(all_evalpoint, axis=0)
        #     all_observed_point = tf.concat(all_observed_point, axis=0)
        #     all_observed_time = tf.concat(all_observed_time, axis=0)
        #     all_generated_samples = tf.concat(all_generated_samples, axis=0)

            # pickle.dump(
            #     [
            #         all_generated_samples,
            #             all_target,
            #             all_evalpoint,
            #             all_observed_point,
            #             all_observed_time,
            #             scaler,
            #             mean_scaler,
            #         ],
            #         f,
            # )

        # CRPS = calc_quantile_CRPS(all_target, all_generated_samples, all_evalpoint, mean_scaler, scaler)

        # with open(f"{path_save}result_nsample" + str(nsample) + ".pk", "wb") as f:
        #     pickle.dump(
        #         [
        #                 np.sqrt(mse_total / evalpoints_total),
        #                 mae_total / evalpoints_total,
        #                 CRPS
        #             ],
        #         f)
        print("RMSE:", np.sqrt(mse_total / evalpoints_total))
        print("MAE:", mae_total / evalpoints_total)
        # print("CRPS:", CRPS)

    return all_generated_samples,all_target


def get_torch_trans(heads=4, layers=1, channels=64):
    return TransformerEncoder(num_layers=layers, units = channels, num_heads = heads, filter_units = 64, reg_lambda = 0)



def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = tf.keras.layers.Conv1D(out_channels, kernel_size,kernel_initializer='he_normal')
    # nn.init.kaiming_normal_(layer.weight)
    return layer


class DiffusionEmbedding(keras.layers.Layer):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        # self.register_buffer(
        #     "embedding",
        #     self._build_embedding(num_steps, embedding_dim / 2),
        #     persistent=False)
        self.embedding = tf.Variable(self._build_embedding(num_steps, embedding_dim / 2))
        self.projection1 = tf.keras.layers.Dense(projection_dim)
        self.projection2 = tf.keras.layers.Dense(projection_dim)

    def call(self, diffusion_step):
        x = tf.gather(self.embedding,diffusion_step.numpy())
        x = self.projection1(x)
        x = tf.nn.silu(x)
        x = self.projection2(x)
        x = tf.nn.silu(x)
        return x

    def _build_embedding(self, num_steps, dim=64):
        steps = tf.expand_dims(tf.range([num_steps]),1)  # (T,1)
        frequencies = 10.0 **  tf.expand_dims(tf.range([dim]) / (dim - 1) * 4.0,0)  # (1,dim)
        table = tf.cast(steps,tf.float32) * frequencies  # (T,dim)
        table = tf.concat([tf.math.sin(table), tf.math.cos(table)], axis=1)  # (T,dim*2)
        return table

    
class diff_CSDI(keras.layers.Layer):
    def __init__(self, config, inputdim=2):
        super().__init__()
        self.channels = config["channels"]

        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config["num_steps"],
            embedding_dim=config["diffusion_embedding_dim"])

        self.input_projection = Conv1d_with_init(inputdim, self.channels, 1)
        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        self.output_projection2 = tf.keras.layers.Conv1D(1, 1,kernel_initializer='zeros')

        self.residual_layers = [
                ResidualBlock(
                    side_dim=config["side_dim"],
                    channels=self.channels,
                    diffusion_embedding_dim=config["diffusion_embedding_dim"],
                    nheads=config["nheads"],
                )
                for _ in range(config["layers"])
            ]

    def call(self, x, cond_info, diffusion_step):
        B, inputdim, K, L = x.shape

        x = tf.reshape(x,[B, inputdim, K * L])
        x = tf.transpose(x,perm=[0,2,1])
        x = self.input_projection(x)
        x = tf.transpose(x, perm=[0, 2, 1])
        x = tf.nn.relu(x)
        x = tf.reshape(x,[B, self.channels, K, L])

        diffusion_emb = self.diffusion_embedding(diffusion_step)

        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, cond_info, diffusion_emb)
            skip.append(skip_connection)

        x = tf.reduce_sum(tf.stack(skip), axis=0) / math.sqrt(len(self.residual_layers))
        x = tf.reshape(x,[B, self.channels, K * L])
        x = tf.transpose(x, perm=[0, 2, 1])
        x = self.output_projection1(x)  # (B,channel,K*L)
        x = tf.nn.relu(x)
        x = self.output_projection2(x)  # (B,1,K*L)
        x = tf.transpose(x, perm=[0, 2, 1])
        x = tf.reshape(x,[B, K, L])
        return x

    
class ResidualBlock(keras.layers.Layer):
    def __init__(self, side_dim, channels, diffusion_embedding_dim, nheads):
        super().__init__()
        self.diffusion_projection = tf.keras.layers.Dense(channels)
        self.cond_projection = Conv1d_with_init(side_dim, 2 * channels, 1)
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)

        self.time_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)
        self.feature_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)

    def forward_time(self, y, base_shape):
        B, channel, K, L = base_shape
        if L == 1:
            return y
        y = tf.reshape(tf.transpose(tf.reshape(y,[B, channel, K, L]),perm=[0, 2, 1, 3]),[B * K, channel, L])
        y = tf.transpose(self.time_layer(tf.transpose(y,perm=[2, 0, 1]))[0],perm=[1, 2, 0])
        y = tf.reshape(tf.transpose(tf.reshape(y,[B, K, channel, L]),perm=[0, 2, 1, 3]),[B, channel, K * L])
        return y

    def forward_feature(self, y, base_shape):
        B, channel, K, L = base_shape
        if K == 1:
            return y
        y = tf.reshape(tf.transpose(tf.reshape(y,[B, channel, K, L]),perm=[0, 3, 1, 2]),[B * L, channel, K])
        y = tf.transpose(self.feature_layer(tf.transpose(y,perm=[2, 0, 1]))[0],perm=[1, 2, 0])
        y = tf.reshape(tf.transpose(tf.reshape(y,[B, L, channel, K]),perm=[0, 2, 3, 1]),[B, channel, K * L])
        return y

    def call(self, x, cond_info, diffusion_emb):
        B, channel, K, L = x.shape
        base_shape = x.shape
        x = tf.reshape(x,[B, channel, K * L])

        diffusion_emb = tf.expand_dims(self.diffusion_projection(diffusion_emb),-1)  # (B,channel,1)
        y = x + diffusion_emb

        y = self.forward_time(y, base_shape)
        y = self.forward_feature(y, base_shape)  # (B,channel,K*L)
        y = tf.transpose(y,perm = [0,2,1])
        y = self.mid_projection(y)  # (B,2*channel,K*L)

        
        
        _, cond_dim, _, _ = cond_info.shape
        cond_info = tf.reshape(cond_info,[B, cond_dim, K * L])
        cond_info = tf.transpose(cond_info, perm=[0, 2, 1])
        cond_info = self.cond_projection(cond_info)  # (B,2*channel,K*L)
        y = y + cond_info
        y = tf.transpose(y, perm=[0, 2, 1])

        
        gate, filter = tf.split(y, 2, axis=1)
        y = tf.math.sigmoid(gate) * tf.math.tanh(filter)  # (B,channel,K*L)
        y = tf.transpose(y, perm=[0, 2, 1])
        y = self.output_projection(y)
        y = tf.transpose(y, perm=[0, 2, 1])

        residual, skip = tf.split(y, 2, axis=1)
        x = tf.reshape(x,base_shape)
        residual = tf.reshape(residual,base_shape)
        skip = tf.reshape(skip,base_shape)
        return (x + residual) / math.sqrt(2.0), skip


class CSDI_base(keras.models.Model):
    def __init__(self, target_dim, config):
        super().__init__()
        # self.device = device
        self.target_dim = target_dim

        self.emb_time_dim = config["model"]["timeemb"]
        self.emb_feature_dim = config["model"]["featureemb"]
        self.is_unconditional = config["model"]["is_unconditional"]
        self.target_strategy = config["model"]["target_strategy"]

        self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim
        if self.is_unconditional == False:
            self.emb_total_dim += 1  # for conditional mask
        self.embed_layer = tf.keras.layers.Embedding(input_dim=self.target_dim, output_dim=self.emb_feature_dim)

        config_diff = config["diffusion"]
        config_diff["side_dim"] = self.emb_total_dim

        input_dim = 1 if self.is_unconditional == True else 2
        self.diffmodel = diff_CSDI(config_diff, input_dim)

        # parameters for diffusion models
        self.num_steps = config_diff["num_steps"]
        if config_diff["schedule"] == "quad":
            self.beta = np.linspace(config_diff["beta_start"] ** 0.5, config_diff["beta_end"] ** 0.5, self.num_steps) ** 2
        elif config_diff["schedule"] == "linear":
            self.beta = np.linspace(config_diff["beta_start"], config_diff["beta_end"], self.num_steps)

        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat)
        self.alpha_torch = tf.expand_dims(tf.expand_dims(tf.convert_to_tensor(self.alpha,tf.float32),1),1)

    def time_embedding(self, pos, d_model=128):
        pe = np.zeros([pos.shape[0], pos.shape[1], d_model])
        position = tf.expand_dims(pos,2)
        div_term = 1 /tf.math.pow(10000.0, tf.range(0, d_model, 2)/ d_model)
        pe[:, :, 0::2] = np.sin(position.numpy() * div_term.numpy())
        pe[:, :, 1::2] = np.cos(position.numpy() * div_term.numpy())
        return tf.convert_to_tensor(pe)

    def get_randmask(self, observed_mask):
        rand_for_mask = np.random.uniform(0,1,observed_mask.shape) * observed_mask.numpy()
        rand_for_mask = np.reshape(rand_for_mask,[len(rand_for_mask),-1])
        for i in range(len(observed_mask)):
            sample_ratio = np.random.rand()  
            num_observed = tf.reduce_sum(observed_mask[i]).numpy()
            num_masked = round(num_observed * sample_ratio)
            idx_to_set = np.argpartition(rand_for_mask[i], -num_masked)[-num_masked:]
            rand_for_mask[i][idx_to_set] = -1
        rand_for_mask = tf.convert_to_tensor(rand_for_mask)
        cond_mask = tf.cast(tf.reshape((rand_for_mask > 0),observed_mask.shape),tf.float32)
        return cond_mask

    def get_hist_mask(self, observed_mask, for_pattern_mask=None):
        if for_pattern_mask is None:
            for_pattern_mask = observed_mask
        if self.target_strategy == "mix":
            rand_mask = self.get_randmask(observed_mask)

        cond_mask = observed_mask.clone()
        for i in range(len(cond_mask)):
            mask_choice = np.random.rand()
            if self.target_strategy == "mix" and mask_choice > 0.5:
                cond_mask[i] = rand_mask[i]
            else: 
                cond_mask[i] = cond_mask[i] * for_pattern_mask[i - 1]
        return cond_mask

    
    
    def get_side_info(self, observed_tp, cond_mask):
        
        B, K, L = cond_mask.shape

        time_embed = self.time_embedding(observed_tp, self.emb_time_dim)  # (B,L,emb)
        time_embed = tf.broadcast_to(tf.expand_dims(time_embed,2),[B, L, K, self.emb_time_dim])
        feature_embed = self.embed_layer(tf.range(self.target_dim))  # (K,B)
        feature_embed = tf.broadcast_to(tf.expand_dims(tf.expand_dims(feature_embed,0),0),[B, L, K, B])

        side_info = tf.concat([tf.cast(time_embed,tf.float32), feature_embed], axis=-1)  # (B,L,K,*)
        side_info = tf.transpose(side_info,perm=[0, 3, 2, 1])  # (B,*,K,L)

        if self.is_unconditional == False:
            side_mask = tf.expand_dims(cond_mask,1)  # (B,1,K,L)
            side_info = tf.concat([side_info, side_mask], axis=1)

        return side_info

    
    def calc_loss_valid(self, observed_data, cond_mask, observed_mask, side_info, is_train):
        loss_sum = 0
        for t in range(self.num_steps):  # calculate loss for all t
            loss = self.calc_loss(observed_data, cond_mask, observed_mask, side_info, is_train, set_t=t)
            loss_sum += loss.detach()
            
        return loss_sum / self.num_steps

    
    def calc_loss(self, observed_data, cond_mask, observed_mask, side_info, is_train, set_t=-1):
        B, K, L = observed_data.shape
        if is_train != 1:  # for validation
            t = (tf.cast(tf.ones(B) * set_t),tf.int64)
        else:
            t = tf.random.uniform([B], 0, self.num_steps,dtype=tf.int32)
        current_alpha = tf.gather(self.alpha_torch,t.numpy())  # (B,1,1)
        noise = tf.random.normal(observed_data.shape)
        noisy_data = (current_alpha ** 0.5) * observed_data + (1.0 - current_alpha) ** 0.5 * noise

        total_input = self.set_input_to_diffmodel(noisy_data, observed_data, cond_mask)

        predicted = self.diffmodel(total_input, side_info, t)  # (B,K,L)

        target_mask = observed_mask - cond_mask
        residual = (noise - predicted) * target_mask
        
        num_eval = tf.reduce_sum(target_mask)
        loss = tf.reduce_sum((residual ** 2)) / (num_eval if num_eval > 0 else 1)

        return loss

    def set_input_to_diffmodel(self, noisy_data, observed_data, cond_mask):
        if self.is_unconditional == True:
            total_input = tf.expand_dims(noisy_data,1)  # (B,1,K,L)
        else:
            cond_obs = tf.expand_dims((cond_mask * observed_data),1)
            noisy_target = tf.expand_dims(((1 - cond_mask) * noisy_data),1)
            total_input = tf.concat([cond_obs, noisy_target], axis=1)  # (B,2,K,L)

        return total_input

    def impute(self, observed_data, cond_mask, side_info, n_samples):
        B, K, L = observed_data.shape
        imputed_samples = tf.zeros([B, n_samples, K, L])

        for i in range(n_samples):
            # generate noisy observation for unconditional model
            if self.is_unconditional == True:
                noisy_obs = observed_data
                noisy_cond_history = []
                for t in range(self.num_steps):
                    noise = tf.random.normal(noisy_obs.shape)
                    noisy_obs = (self.alpha_hat[t] ** 0.5) * noisy_obs + self.beta[t] ** 0.5 * noise
                    noisy_cond_history.append(noisy_obs * cond_mask)

            current_sample = tf.random.normal(observed_data.shape)

            for t in range(self.num_steps - 1, -1, -1):
                if self.is_unconditional == True:
                    diff_input = cond_mask * noisy_cond_history[t] + (1.0 - cond_mask) * current_sample
                    diff_input = tf.expand_dims(diff_input,1)  # (B,1,K,L)
                else:
                    cond_obs = tf.expand_dims((cond_mask * observed_data),1)
                    noisy_target = tf.expand_dims(((1 - cond_mask) * current_sample),1)
                    diff_input = tf.concat([cond_obs, noisy_target], axis=1)  # (B,2,K,L)
                predicted = self.diffmodel(diff_input, side_info, tf.convert_to_tensor([t]))

                coeff1 = 1 / self.alpha_hat[t] ** 0.5
                coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
                current_sample = coeff1 * (current_sample - coeff2 * predicted)

                if t > 0:
                    noise = tf.random.normal(current_sample.shape)
                    sigma = (
                                    (1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]
                            ) ** 0.5
                    current_sample += sigma * noise
            imputed_samples_numpy = imputed_samples.numpy()
            imputed_samples_numpy[:, i] = current_sample.numpy()

            
        return imputed_samples_numpy

    
    def call(self, batch, is_train=1):
        (observed_data,observed_mask,observed_tp,gt_mask,for_pattern_mask,_) = self.process_data(batch)
        if is_train == 0:
            cond_mask = gt_mask
        elif self.target_strategy != "random":
            cond_mask = self.get_hist_mask(observed_mask, for_pattern_mask=for_pattern_mask)
        else:
            cond_mask = self.get_randmask(observed_mask)
        side_info = self.get_side_info(observed_tp, cond_mask)
        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid

        return loss_func(observed_data, cond_mask, observed_mask, side_info, is_train)

    def evaluate(self, batch, n_samples):
        (observed_data,observed_mask,observed_tp,gt_mask,_,cut_length) = self.process_data(batch)
        cond_mask = gt_mask
        target_mask = observed_mask - cond_mask
        side_info = self.get_side_info(observed_tp, cond_mask)
        samples = self.impute(observed_data, cond_mask, side_info, n_samples)
        target_mask = target_mask.numpy()
        for i in range(len(cut_length)):
            target_mask[i, ..., 0: cut_length[i].numpy()] = 0
                
        return samples, observed_data, target_mask, observed_mask, observed_tp

    
class CSDI_Custom(CSDI_base):
    def __init__(self, config, target_dim=35):
        super(CSDI_Custom, self).__init__(target_dim, config)

    def process_data(self, batch):
        observed_data, observed_mask, gt_mask, observed_tp = batch
        observed_data = tf.cast(observed_data,tf.float32)
        observed_mask = tf.cast(observed_mask,tf.float32)
        observed_tp = tf.cast(observed_tp,tf.float32)
        gt_mask = tf.cast(gt_mask,tf.float32)

        observed_data = tf.transpose(observed_data,perm = [0, 2, 1])
        observed_mask = tf.transpose(observed_mask,perm = [0, 2, 1])
        gt_mask = tf.transpose(gt_mask,perm = [0, 2, 1])

        cut_length = tf.cast(tf.zeros(len(observed_data)),tf.int64)
        for_pattern_mask = observed_mask

        return (observed_data,observed_mask,observed_tp,gt_mask,for_pattern_mask,cut_length)
    
    
def mask_missing_train_rm(data, missing_ratio=0.0):
    observed_values = np.array(data)
    observed_masks = ~np.isnan(observed_values)

    masks = observed_masks.reshape(-1).copy()
    obs_indices = np.where(masks)[0].tolist()
    miss_indices = np.random.choice(obs_indices, int(len(obs_indices) * missing_ratio), replace=False)
    masks[miss_indices] = False
    gt_masks = masks.reshape(observed_masks.shape)
    observed_values = np.nan_to_num(observed_values)
    observed_masks = observed_masks.astype("float32")
    gt_masks = gt_masks.astype("float32")

    return observed_values, observed_masks, gt_masks


def mask_missing_train_nrm(data, k_segments=5):
    observed_values = np.array(data)
    observed_masks = ~np.isnan(observed_values)
    gt_masks = observed_masks.copy()
    length_index = np.array(range(data.shape[0]))
    list_of_segments_index = np.array_split(length_index, k_segments)

    for channel in range(gt_masks.shape[1]):
        s_nan = random.choice(list_of_segments_index)
        gt_masks[:, channel][s_nan[0]:s_nan[-1] + 1] = 0

    observed_values = np.nan_to_num(observed_values)
    observed_masks = observed_masks.astype("float32")
    gt_masks = gt_masks.astype("float32")

    return observed_values, observed_masks, gt_masks


def mask_missing_train_bm(data, missing_ratio):
    observed_values = np.array(data)
    observed_masks = ~np.isnan(observed_values)
    gt_masks = observed_masks.copy()
    length_index = np.array(range(data.shape[0]))
    np.random.shuffle(length_index)
    idx = length_index[0: int(len(length_index) * missing_ratio)]
    gt_masks[:, :29][idx] = 0
    np.random.shuffle(length_index)
    idx = length_index[0: int(len(length_index) * missing_ratio)]
    gt_masks[:, 29:75][idx] = 0
    np.random.shuffle(length_index)
    idx = length_index[0: int(len(length_index) * missing_ratio)]
    gt_masks[:, 75:][idx] = 0

    observed_values = np.nan_to_num(observed_values)
    observed_masks = observed_masks.astype("float32")
    gt_masks = gt_masks.astype("float32")

    return observed_values, observed_masks, gt_masks


def mask_missing_impute(data, mask):
    
    observed_values = np.array(data)
    observed_masks = ~np.isnan(observed_values)
    
    observed_values = np.nan_to_num(observed_values)
    observed_masks = observed_masks.astype("float32")
    mask = mask.astype("float32")
    gt_masks = observed_masks * mask

    return observed_values, observed_masks, gt_masks


class Custom_Train_Dataset():
    def __init__(self, series, path, missing_ratio_or_k=0.0, masking='rm', ms=None):
        self.series = series
        self.length = series.shape[1]
        self.n_channels = series.shape[2]

        self.observed_values = []
        self.observed_masks = []
        self.gt_masks = []

        if not os.path.isfile(path):  # if datasetfile is none, create
            for sample in series:
                if masking == 'rm':
                    sample = tf.stop_gradient(sample).cpu().numpy()
                    observed_values, observed_masks, gt_masks = mask_missing_train_rm(sample, missing_ratio_or_k)
                    observed_values =  tf.convert_to_tensor(observed_values) 
                    observed_masks =  tf.convert_to_tensor(observed_masks) 
                    gt_masks =  tf.convert_to_tensor(gt_masks) 
                elif masking == 'nrm':
                    sample = tf.stop_gradient(sample).cpu().numpy()
                    observed_values, observed_masks, gt_masks = mask_missing_train_nrm(sample, missing_ratio_or_k)
                    observed_values =  tf.convert_to_tensor(observed_values) 
                    observed_masks =  tf.convert_to_tensor(observed_masks) 
                    gt_masks =  tf.convert_to_tensor(gt_masks) 
                elif masking == 'bm':
                    sample = tf.stop_gradient(sample).cpu().numpy()
                    observed_values, observed_masks, gt_masks = mask_missing_train_bm(sample, missing_ratio_or_k)
                    observed_values =  tf.convert_to_tensor(observed_values) 
                    observed_masks =  tf.convert_to_tensor(observed_masks) 
                    gt_masks =  tf.convert_to_tensor(gt_masks) 
                    
                self.observed_values.append(observed_values)
                self.observed_masks.append(observed_masks)
                self.gt_masks.append(gt_masks)
                

        self.use_index_list = np.arange(len(self.observed_values))

    # def __getitem__(self, org_index):
    #     index = self.use_index_list[org_index]
    #     s = {
    #         "observed_data": self.observed_values[index],
    #         "observed_mask": self.observed_masks[index],
    #         "gt_mask": self.gt_masks[index],
    #         "timepoints": np.arange(self.length),
    #     }
    #     return s

    # def __len__(self):
    #     return len(self.use_index_list)

    
class Custom_Impute_Dataset():
    def __init__(self, series, masking, use_index_list=None, path = ''):
        self.series = series
        self.n_channels = series.shape[2]
        self.length = series.shape[1]
        self.mask = masking

        self.observed_values = []
        self.observed_masks = []
        self.gt_masks = []

        missing_ratio_or_k = 0.05

        # if not os.path.isfile(path):  # if datasetfile is none, create
        for sample in series:
            if masking == 'rm':
                sample = tf.stop_gradient(sample).cpu().numpy()
                observed_values, observed_masks, gt_masks = mask_missing_train_rm(sample, missing_ratio_or_k)
                observed_values =  tf.convert_to_tensor(observed_values)
                observed_masks =  tf.convert_to_tensor(observed_masks)
                gt_masks =  tf.convert_to_tensor(gt_masks)
            elif masking == 'nrm':
                sample = tf.stop_gradient(sample).cpu().numpy()
                observed_values, observed_masks, gt_masks = mask_missing_train_nrm(sample, missing_ratio_or_k)
                observed_values =  tf.convert_to_tensor(observed_values)
                observed_masks =  tf.convert_to_tensor(observed_masks)
                gt_masks =  tf.convert_to_tensor(gt_masks)
            elif masking == 'bm':
                sample = sample.detach().cpu().numpy()
                observed_values, observed_masks, gt_masks = mask_missing_train_bm(sample, missing_ratio_or_k)
                observed_values =  tf.convert_to_tensor(observed_values)
                observed_masks =  tf.convert_to_tensor(observed_masks)
                gt_masks =  tf.convert_to_tensor(gt_masks)

                #observed_values, observed_masks, gt_masks = mask_missing_impute(sample, mask)

            self.observed_values.append(sample)
            self.observed_masks.append(observed_masks)
            self.gt_masks.append(gt_masks)

                
        if use_index_list is None:
            self.use_index_list = np.arange(len(self.observed_values))
        else:
            self.use_index_list = use_index_list

    def __getitem__(self, org_index):
        index = self.use_index_list[org_index]
        s = {
            "observed_data": self.observed_values[index],
            "observed_mask": self.observed_masks[index],
            "gt_mask": self.gt_masks[index],
            "timepoints": np.arange(self.length),
        }
        return s

    def __len__(self):
        return len(self.use_index_list)
    
    
def get_dataloader_train_impute(series,
                                batch_size=4,
                                missing_ratio_or_k=0.1,
                                train_split=0.7,
                                valid_split=0.9,
                                len_dataset=100,
                                masking='rm',
                               path_save='',
                               ms=None):


    train_dataset = Custom_Train_Dataset(series=series,
                                         missing_ratio_or_k=missing_ratio_or_k,
                                         masking=masking, path=path_save, ms=1)
    train_loader = train_dataset
    return train_loader


def get_dataloader_impute(series, mask, batch_size=4, len_dataset=100):
    indlist = np.arange(len_dataset)
    impute_dataset = Custom_Impute_Dataset(series=series, use_index_list=indlist,masking=mask)
    # impute_loader = DataLoader(impute_dataset, batch_size=batch_size, shuffle=False)

    return impute_dataset



class CSDIImputer:
    def __init__(self):
        np.random.seed(0)
        random.seed(0)
        
        '''
        CSDI imputer
        3 main functions:
        a) training based on random missing, non-random missing, and blackout masking.
        b) loading weights of already trained model
        c) impute samples in inference. Note, you must manually load weights after training for inference.
        '''

    def train(self,
              series,
              masking ='rm',
              missing_ratio_or_k = 0.0,
              path_save="",
              train_split = 0.7,
              valid_split = 0.2,
              epochs = 200,
              samples_generate = 10,
              batch_size = 16,
              lr = 1.0e-3,
              layers = 4,
              channels = 64,
              nheads = 8,
              difussion_embedding_dim = 128,
              beta_start = 0.0001,
              beta_end = 0.5,
              num_steps = 50,
              schedule = 'quad',
              is_unconditional = 0,
              timeemb = 128,
              featureemb = 16,
              target_strategy = 'random',
             ):
        
        '''
        CSDI training function. 
       
       
        Requiered parameters
        -series: Assumes series of shape (Samples, Length, Channels).
        -masking: 'rm': random missing, 'nrm': non-random missing, 'bm': black-out missing.
        -missing_ratio_or_k: missing ratio 0 to 1 for 'rm' masking and k segments for 'nrm' and 'bm'.
        -path_save: full path where to save model weights, configuration file, and means and std devs for de-standardization in inference.
        
        Default parameters
        -train_split: 0 to 1 representing the percentage of train set from whole data.
        -valid_split: 0 to 1. Is an adition to train split where 1 - train_split - valid_split = test_split (implicit in method).
        -epochs: number of epochs to train.
        -samples_generate: number of samples to be generated.
        -batch_size: batch size in training.
        -lr: learning rate.
        -layers: difussion layers.
        -channels: number of difussion channels.
        -nheads: number of difussion 'heads'.
        -difussion_embedding_dim: difussion embedding dimmensions. 
        -beta_start: start noise rate.
        -beta_end: end noise rate.
        -num_steps: number of steps.
        -schedule: scheduler. 
        -is_unconditional: conditional or un-conditional imputation. Boolean.
        -timeemb: temporal embedding dimmensions.
        -featureemb: feature embedding dimmensions.
        -target_strategy: strategy of masking. 
        -wandbiases_project: weight and biases project.
        -wandbiases_experiment: weight and biases experiment or run.
        -wandbiases_entity: weight and biases entity. 
        '''
       
        config = {}
        
        config['train'] = {}
        config['train']['epochs'] = epochs
        config['train']['batch_size'] = batch_size
        config['train']['lr'] = lr
        config['train']['train_split'] = train_split
        config['train']['valid_split'] = valid_split
        config['train']['path_save'] = path_save
        
       
        config['diffusion'] = {}
        config['diffusion']['layers'] = layers
        config['diffusion']['channels'] = channels
        config['diffusion']['nheads'] = nheads
        config['diffusion']['diffusion_embedding_dim'] = difussion_embedding_dim
        config['diffusion']['beta_start'] = beta_start
        config['diffusion']['beta_end'] = beta_end
        config['diffusion']['num_steps'] = num_steps
        config['diffusion']['schedule'] = schedule
        
        config['model'] = {} 
        config['model']['missing_ratio_or_k'] = missing_ratio_or_k
        config['model']['is_unconditional'] = is_unconditional
        config['model']['timeemb'] = timeemb
        config['model']['featureemb'] = featureemb
        config['model']['target_strategy'] = target_strategy
        config['model']['masking'] = masking
        
        print(json.dumps(config, indent=4))

        config_filename = path_save + "config_csdi_training"
        print('configuration file name:', config_filename)
        # with open('temp' + ".json", "w") as f:
        #     json.dump(config, f, indent=4)


        train_loader = get_dataloader_train_impute(
            series=series,
            train_split=config["train"]["train_split"],
            valid_split=config["train"]["valid_split"],
            len_dataset=series.shape[0],
            batch_size=config["train"]["batch_size"],
            missing_ratio_or_k=config["model"]["missing_ratio_or_k"],
            masking=config['model']['masking'],
            path_save=config['train']['path_save'])

        model = CSDI_Custom(config, target_dim=series.shape[2])

        train(model=model,
              config=config["train"],
              train_loader=train_loader,
              path_save=config['train']['path_save'])

        evaluate(model=model,
                 test_loader=test_loader,
                 nsample=samples_generate,
                 scaler=1,
                 path_save=config['train']['path_save'])
        
        
    def load_weights(self, 
                     path_load_model='',
                     path_config=''):
        
        self.path_load_model_dic = path_load_model
        self.path_config = path_config
    
    
        '''
        Load weights and configuration file for inference.
        
        path_load_model: load model weights
        path_config: load configuration file
        '''
    

    def impute(self,
               sample,
               mask,
               n_samples = 50,
               ):
        
        '''
        Imputation function 
        sample: sample(s) to be imputed (Samples, Length, Channel)
        mask: mask where values to be imputed. 0's to impute, 1's to remain. 
        n_samples: number of samples to be generated
        return imputations with shape (Samples, N imputed samples, Length, Channel)
        '''
        
        if len(sample.shape) == 2:
            self.series_impute =  tf.convert_to_tensor(np.expand_dims(sample, axis=0))
        elif len(sample.shape) == 3:
            self.series_impute = sample

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

        samples_generate = 10
        batch_size = 16
        missing_ratio_or_k = 0.05
        lr = 1.0e-3
        layers = 4
        channels = 64
        nheads = 8
        difussion_embedding_dim = 128
        beta_start = 0.0001
        beta_end = 0.5
        num_steps = 50
        schedule = 'quad'
        is_unconditional = 0
        timeemb = 128
        featureemb = 16
        target_strategy = 'random'

        config['diffusion'] = {}
        config['diffusion']['layers'] = layers
        config['diffusion']['channels'] = channels
        config['diffusion']['nheads'] = nheads
        config['diffusion']['diffusion_embedding_dim'] = difussion_embedding_dim
        config['diffusion']['beta_start'] = beta_start
        config['diffusion']['beta_end'] = beta_end
        config['diffusion']['num_steps'] = num_steps
        config['diffusion']['schedule'] = schedule

        config['model'] = {}
        config['model']['missing_ratio_or_k'] = missing_ratio_or_k
        config['model']['is_unconditional'] = is_unconditional
        config['model']['timeemb'] = timeemb
        config['model']['featureemb'] = featureemb
        config['model']['target_strategy'] = target_strategy
        config['model']['masking'] = mask

        test_loader = get_dataloader_impute(series=self.series_impute,len_dataset=len(self.series_impute),
                                            mask=mask, batch_size=16)

        model = CSDI_Custom(config, target_dim=self.series_impute.shape[2])

        epoch_no = 120
        checkpoint_name = '{}CSDI.pkl'.format(epoch_no)
        model_path = os.path.join(config["train_config"]['output_directory'], checkpoint_name)
        model.load_weights(model_path)

        imputations, targets = evaluate(model=model,
                                test_loader=test_loader,
                                nsample=n_samples,
                                scaler=1,
                                path_save='')
        
        # indx_imputation = ~mask.astype(bool)
        #
        # original_sample_replaced =[]
        #
        # for original_sample, single_n_samples in zip(sample.numpy(), imputations): # [x,x,x] -> [x,x] & [x,x,x,x] -> [x,x,x]
        #     single_sample_replaced = []
        #     for sample_generated in single_n_samples:  # [x,x] & [x,x,x] -> [x,x]
        #         sample_out = original_sample.copy()
        #         sample_out[indx_imputation] = sample_generated[indx_imputation]
        #         single_sample_replaced.append(sample_out)
        #     original_sample_replaced.append(single_sample_replaced)
            
        # output = np.array(original_sample_replaced)
        
        
        return imputations, targets


def get_lr_values(initial,bounds,gamma):
    values = [initial]
    for i in range(len(bounds)):
        initial = initial * gamma
        values.append(initial)
    return values

def cut(obj, sec):
    return [obj[i:i+sec] for i in range(0,len(obj),sec)]
    


