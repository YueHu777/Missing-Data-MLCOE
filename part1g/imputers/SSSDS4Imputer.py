import math
import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
from utils.util import calc_diffusion_step_embedding,time_embedding
from imputers.S4Model import S4Layer
import numpy as np



class Conv(keras.layers.Layer):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super(Conv, self).__init__()
        self.conv = tf.keras.layers.Conv1D(out_channels, kernel_size,dilation_rate=dilation, padding='same',kernel_initializer='he_normal')


    def call(self, x):
        out = self.conv(x)
        return out


    
class ZeroConv1d(keras.layers.Layer):
    def __init__(self, in_channel, out_channel):
        super(ZeroConv1d, self).__init__()
        self.conv = tf.keras.layers.Conv1D(out_channel, kernel_size=1, padding='valid', bias_initializer='zeros',kernel_initializer='zeros')


    def call(self, x):
        out = self.conv(x)
        return out


class Residual_block(keras.layers.Layer):
    def __init__(self, res_channels, skip_channels, 
                 diffusion_step_embed_dim_out, in_channels,
                s4_lmax,
                s4_d_state,
                s4_dropout,
                s4_bidirectional,
                s4_layernorm):
        super(Residual_block, self).__init__()
        self.res_channels = res_channels
        self.BN =tf.keras.layers.BatchNormalization()

        self.fc_t = tf.keras.layers.Dense(self.res_channels, activation=None)
        
        self.S41 = S4Layer(features=2*self.res_channels, 
                          lmax=s4_lmax,
                          N=s4_d_state,
                          dropout=s4_dropout,
                          bidirectional=s4_bidirectional,
                          layer_norm=s4_layernorm)
 
        self.conv_layer = Conv(self.res_channels, 2 * self.res_channels, kernel_size=3)

        self.S42 = S4Layer(features=2*self.res_channels, 
                          lmax=s4_lmax,
                          N=s4_d_state,
                          dropout=s4_dropout,
                          bidirectional=s4_bidirectional,
                          layer_norm=s4_layernorm)
        
        self.cond_conv = Conv(2*in_channels, 2*self.res_channels, kernel_size=1)

        self.res_conv = tf.keras.layers.Conv1D(res_channels, kernel_size=1,kernel_initializer='he_normal')

        self.skip_conv = tf.keras.layers.Conv1D(skip_channels, kernel_size=1, kernel_initializer='he_normal')


    def call(self, input_data):
        x, cond, diffusion_step_embed = input_data
        h = x
        B, L, C = x.shape
        assert C == self.res_channels                      
                 
        part_t = self.fc_t(diffusion_step_embed)
        part_t = tf.reshape(part_t,[B, self.res_channels, 1])
        part_t = tf.transpose(part_t, perm=[0, 2, 1])
        h = h + part_t

        h = self.conv_layer(h)
        h = tf.transpose(self.S41(tf.transpose(h,perm=[1,0,2])),perm=[1,0,2])
        
        assert cond is not None
        cond = self.cond_conv(cond)
        h += cond
        
        h = tf.transpose(self.S42(tf.transpose(h,perm=[1,0,2])),perm=[1,0,2])

        h = tf.transpose(h, perm=[0, 2, 1])
        out = tf.math.tanh(h[:,:self.res_channels,:]) * tf.math.sigmoid(h[:,self.res_channels:,:])
        out = tf.transpose(out, perm=[0, 2, 1])
        res = self.res_conv(out)

        assert x.shape == res.shape
        skip = self.skip_conv(out)

        return (x + res) * math.sqrt(0.5), skip  # normalize for training stability


class Residual_group(keras.layers.Layer):
    def __init__(self, res_channels, skip_channels, num_res_layers, 
                 diffusion_step_embed_dim_in, 
                 diffusion_step_embed_dim_mid,
                 diffusion_step_embed_dim_out,
                 in_channels,
                 s4_lmax,
                 s4_d_state,
                 s4_dropout,
                 s4_bidirectional,
                 s4_layernorm):
        super(Residual_group, self).__init__()
        self.num_res_layers = num_res_layers
        self.diffusion_step_embed_dim_in = diffusion_step_embed_dim_in

        self.fc_t1 = tf.keras.layers.Dense(diffusion_step_embed_dim_mid,activation=None)
        self.fc_t2 = tf.keras.layers.Dense(diffusion_step_embed_dim_out,activation=None)
        
        self.residual_blocks = []

        for n in range(self.num_res_layers):
            self.residual_blocks.append(Residual_block(res_channels, skip_channels, 
                                                       diffusion_step_embed_dim_out=diffusion_step_embed_dim_out,
                                                       in_channels=in_channels,
                                                       s4_lmax=s4_lmax,
                                                       s4_d_state=s4_d_state,
                                                       s4_dropout=s4_dropout,
                                                       s4_bidirectional=s4_bidirectional,
                                                       s4_layernorm=s4_layernorm))

            
    def call(self, input_data):
        noise, conditional, diffusion_steps = input_data

        diffusion_step_embed = calc_diffusion_step_embedding(diffusion_steps, self.diffusion_step_embed_dim_in)
        diffusion_step_embed = keras.activations.swish(self.fc_t1(diffusion_step_embed))
        diffusion_step_embed = keras.activations.swish(self.fc_t2(diffusion_step_embed))

        h = noise
        skip = 0
        for n in range(self.num_res_layers):
            h, skip_n = self.residual_blocks[n]((h, conditional, diffusion_step_embed))  
            skip += skip_n  

        return skip * math.sqrt(1.0 / self.num_res_layers)


class SSSDS4Imputer(keras.models.Model):
    def __init__(self, in_channels, res_channels, skip_channels, out_channels, 
                 num_res_layers,
                 diffusion_step_embed_dim_in, 
                 diffusion_step_embed_dim_mid,
                 diffusion_step_embed_dim_out,
                 s4_lmax,
                 s4_d_state,
                 s4_dropout,
                 s4_bidirectional,
                 s4_layernorm):
        super(SSSDS4Imputer, self).__init__()

        self.init_conv =tf.keras.Sequential([Conv(in_channels, res_channels,kernel_size=10),tf.keras.layers.ReLU()])

        self.residual_layer = Residual_group(res_channels=res_channels, 
                                             skip_channels=skip_channels, 
                                             num_res_layers=num_res_layers, 
                                             diffusion_step_embed_dim_in=diffusion_step_embed_dim_in,
                                             diffusion_step_embed_dim_mid=diffusion_step_embed_dim_mid,
                                             diffusion_step_embed_dim_out=diffusion_step_embed_dim_out,
                                             in_channels=in_channels,
                                             s4_lmax=s4_lmax,
                                             s4_d_state=s4_d_state,
                                             s4_dropout=s4_dropout,
                                             s4_bidirectional=s4_bidirectional,
                                             s4_layernorm=s4_layernorm)
        
        self.final_conv = tf.keras.Sequential([Conv(skip_channels, skip_channels, kernel_size=10),
                                        tf.keras.layers.ReLU(),
                                        ZeroConv1d(skip_channels, out_channels)])

    def call(self, input_data):
        
        noise, conditional, mask, diffusion_steps,observed_mask = input_data

        conditional = conditional * mask
        timepoints = tf.convert_to_tensor(np.tile(np.arange(conditional.shape[1]), [16, 1]))
        side_info = time_embedding(timepoints)
        conditional = tf.concat([conditional, tf.cast(mask,tf.float32),tf.cast(side_info,tf.float32)], axis=2)

        x = noise
        x = self.init_conv(x)
        x = self.residual_layer((x, conditional, diffusion_steps))
        y = self.final_conv(x)
        return y
