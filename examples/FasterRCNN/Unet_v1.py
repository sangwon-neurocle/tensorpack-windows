# -*- coding: utf-8 -*-
# Author: Your Name <your@email.com>

import argparse
import os
import tensorflow as tf
import tensorflow.contrib as tc
import numpy as np
import math
import sys
sys.path.append('.')
import re
# from tensorpack import *
from dataset import register_coco, register_balloon
from config import config as cfg
from config import finalize_configs
from data import get_train_dataflow
from eval import EvalCallback
from tensorpack.tfutils import optimizer
from tensorpack import ModelDesc
from tensorpack.tfutils.summary import add_moving_summary

"""
This is a boiler-plate template.
All code is in this file is the most minimalistic way to solve a deep-learning problem with cross-validation.
"""

BATCH_SIZE = 4
SHAPE = 512
CHANNELS = 3

def padx1tox2andconcat(x1, x2):
    assert(x1.shape[-1]==x2.shape[-1]), f"different number of channel to concat: {x1.shape[-1]}, {x2.shape[-1]}"

    diffX = x2.shape[1] - x1.shape[1]
    diffY = x2.shape[2] - x1.shape[2]
    padding = [[0,0], [diffY//2, diffY - diffY//2], [diffX//2, diffX - diffX//2], [0,0]]
    x1 = tf.pad(x1, padding)
    x1x2 = tf.concat([x1,x2], -1)
    return x1x2

class Up():
    """ Upscaling then double conv """
    def __init__(self, in_channels, out_channels, bn_params, is_training):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bn_params = bn_params
        self.is_training = is_training
        self.conv = DoubleConv(self.in_channels, self.out_channels, self.bn_params, self.is_training)._build_layer
    def _build_layer(self, x1, x2):
        self.x1 = x1
        self.x2 = x2
        # assert(x1.shape[-1]==x2.shape[-1]), f"different number of channel : {x1.shape[-1]}, {x2.shape[-1]}"
        print(f"before upsampling {x1.shape}")
        while x1.shape[1]*2 <= x2.shape[1] :
            # x1 = tf.layers.Conv2DTranspose(x2.shape[-1], kernel_size = 2, strides = 2, padding='same').apply(x1)
            x1 = tc.layers.conv2d_transpose(x1, self.in_channels//2, kernel_size = 2, stride = 2, padding='same')
        # if not self.isFinal:
        #     x1 = tc.layers.conv2d_transpose(x1, self.in_channels//2, kernel_size = 4, stride = 2, padding='same')
        # else:
        #     x1 = tc.layers.conv2d_transpose(x1, self.in_channels//2, kernel_size = 4, stride = 2, padding='same')
        #     x1 = tc.layers.conv2d_transpose(x1, self.in_channels//2, kernel_size = 4, stride = 2, padding='same')
        print(f"after upsampling {x1.shape}")

        # before upsampling (?, 8, 8, 1024)
        # after upsampling (?, 16, 16, 512)

        # before upsampling (?, 16, 16, 256)
        # after upsampling (?, 32, 32, 256)

        # before upsampling (?, 64, 64, 64)
        # after upsampling (?, 128, 128, 64)


        # x1 = DoubleConv(self.out_channels, self.in_channels//2)._build_layer(x1)

        x1x2 = padx1tox2andconcat(x1,x2)
        print(f"x1.shape {x1.shape} x2.shape {x2.shape}")
        print(f"x1x2.shape {x1x2.shape}")
        # x1.shape (?, 16, 16, 256) x2.shape (?, 16, 16, 512)
        # x1x2.shape (?, 16, 16, 768)

        x3 = self.conv(x1x2)
        
        print(f"x3.shape {x3.shape}")

        return x3

class DoubleConv():
    """ (conv => BN => ReLU) * 2 """   
    def __init__(self, in_channels, out_channels, bn_params, is_training):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bn_params = bn_params
        self.normalizer = tc.layers.batch_norm
        self.is_training = is_training
    def _build_layer(self, x):
        self.x = x
        assert (x.shape[-1]==self.in_channels), f"Expect {self.in_channels} channels but # of input channel is {x.shape[-1]}"
        # conv2d
        output = tc.layers.conv2d(self.x, self.out_channels, kernel_size = 3, padding='same')
        # Batch Normalization
        output = tc.layers.batch_norm(output, is_training=self.is_training)
        # RelU
        output = tf.nn.relu(output)
        # # conv2d + BN + RelU
        # output = tc.layers.conv2d(self.x, self.out_channels, kernel_size = 3, padding='same', 
        #     activation_fn=tf.nn.relu6, data_format=self.data_format, normalizer_fn=self.normalizer, normalizer_params=self.bn_params)
        # conv2d
        output = tc.layers.conv2d(output, self.out_channels, kernel_size = 3, padding='same')
        # Batch Normalization
        output = tc.layers.batch_norm(output, is_training=self.is_training)
        # RelU
        output = tf.nn.relu(output)
        # # conv2d + BN + RelU
        # output = tc.layers.conv2d(self.x, self.out_channels, kernel_size = 3, padding='same', 
        #     activation_fn=tf.nn.relu6, data_format=self.data_format, normalizer_fn=self.normalizer, normalizer_params=self.bn_params)
        
        return output

class Out1x1Conv():
    """ Upscaling then double conv """
    def __init__(self, in_channels, out_channels):
        self.in_channels = in_channels
        self.out_channels = out_channels
    def _build_layer(self, x):
        self.x = x
        output = tc.layers.conv2d(self.x, self.out_channels, kernel_size = 1, padding='same')
        # print(f"output shape : {output.shape}")
        return output

class Unet(ModelDesc):
    def inputs(self):
        return [tf.TensorSpec((None, SHAPE, SHAPE, CHANNELS), tf.float32, 'inputx'),
                tf.TensorSpec((None, SHAPE, SHAPE), tf.float32, 'inputy'),
                tf.TensorSpec((None,), tf.int32, 'num_classes')]

    def build_graph(self, inputx, inputy, num_classes):
        ret_dict = {}
        base_dict = get_base_dict(inputx)
        convs = []
        backbone_channels = []
        for idx, fname in enumerate(base_dict["convfeats"]):
            feat = base_dict[fname]
            # log.debug(f'idx : {idx} | fname : {fname} | feat : {feat}')
            convs.append(feat)
            backbone_channels.append(feat.shape.as_list()[-1]) 
        
        just_before_output_channel = backbone_channels[0] // 2
        x1 = DoubleConv(self.X.shape[-1],just_before_output_channel, self.bn_params, self.is_train)._build_layer(self.X)
        
        prev = convs[-1]
        for i in range(len(backbone_channels)-1, 0, -1):
            # i : len(backbone_channels)-1 ~ 1
            print(f"i : {i}")
            print(f"backbone_channels[i] : {backbone_channels[i]}")
            channel_from = backbone_channels[i]
            channel_to = backbone_channels[i-1]//2
            with tf.variable_scope('layer{}'.format(i+1)):
                ret_dict['up{}_{}'.format(channel_from, channel_to)] = prev = Up(channel_from, channel_to, self.bn_params, self.is_train)._build_layer(prev, convs[i-1])

        with tf.variable_scope('layer1'):
            ret_dict['up{}_{}'.format(backbone_channels[0], just_before_output_channel)] = prev = Up(backbone_channels[0],just_before_output_channel, self.bn_params, self.is_train)._build_layer(prev, x1) # original
        with tf.variable_scope('output_layer'):
            ret_dict['logits'] = Out1x1Conv(just_before_output_channel, self.class_num)._build_layer(prev)

        y_pred = ret_dict['logits']
        y_true = inputy

        # loss
        def dice_loss(y_true, y_pred):
            smooth = 1e-6
            # Flatten
            # y_true_f = K.flatten(y_true)
            # y_pred_f = K.flatten(y_pred)
            y_true = tf.one_hot(y_true,self.class_num,axis=0)
            y_true = tf.transpose(y_true, perm=(1, 2, 3, 0),)
            log.debug(f"newinputy.shape : {y_true.shape}")

            y_pred = tf.nn.softmax(y_pred)

            y_true_f = tf.reshape(y_true, [-1])
            y_pred_f = tf.reshape(y_pred, [-1])
            intersection = y_true_f * y_pred_f
            score = (2. * tf.reduce_sum(intersection) + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
            return 1. - score

        def bce_dice_loss(y_true, y_pred):
            softmax_losses =tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true )
            softmax_loss = tf.reduce_mean(softmax_losses)
            return softmax_loss + dice_loss(y_true, y_pred)

        cost = bce_dice_loss(y_true, y_pred)
        summary.add_moving_summary(cost)
        return cost

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=5e-3, trainable=False)
        return tf.train.AdamOptimizer(lr)

def get_base_dict(inputx):
    # Sconv7Net_v2 -> _build_model(self, inputx, hparamd)
    ret_dict = {}

    multp = 1.0
    
    initial_conv_channel_sz = max(8, int(32*self.depth_multiplier//8 * 8))

    current_conv_channel_sz = initial_conv_channel_sz
    maxpool_ksz = [1, 2, 2, 1]
    maxpool_strides = [1, 2, 2, 1]
    if self.data_format == 'NCHW':
        maxpool_ksz = [1, 1, 2, 2]
        maxpool_strides = [1, 1, 2, 2]
    
    # print("input d1 {}".format(inputx))
    output = tc.layers.conv2d(inputx, current_conv_channel_sz, 3, 1, data_format = self.data_format,
                                    normalizer_fn=self.normalizer, normalizer_params=self.bn_params)                              
    output = tf.identity(output, name = 'conv_224_endp')
    # print("input d2 {}".format(output))
    output = tf.nn.max_pool(output, ksize = maxpool_ksz, strides = maxpool_strides, padding='SAME', data_format = self.data_format)
    current_conv_channel_sz *= 2 
    output = tc.layers.conv2d(output, current_conv_channel_sz, 3, 1, data_format = self.data_format,
                                    normalizer_fn=self.normalizer, normalizer_params=self.bn_params)
    output = tf.identity(output, name = 'conv_112_0_endp')
    # print("input d3 {}".format(output))
    output = tf.nn.max_pool(output, ksize = maxpool_ksz, strides = maxpool_strides, padding='SAME', data_format = self.data_format)
    current_conv_channel_sz *= 2 
    output = tc.layers.conv2d(output, current_conv_channel_sz, 3, 1, data_format = self.data_format,
                                    normalizer_fn=self.normalizer, normalizer_params=self.bn_params)
    output = tf.identity(output, name = 'conv_56_0_endp')
    ret_dict['conv_56_0_endp'] = output 
    # print("input d4 {}".format(output))
    output = tf.nn.max_pool(output, ksize = maxpool_ksz, strides = maxpool_strides, padding='SAME', data_format = self.data_format)
    current_conv_channel_sz *= 2 
    output = tc.layers.conv2d(output, current_conv_channel_sz, 3, 1, data_format = self.data_format,
                                    normalizer_fn=self.normalizer, normalizer_params=self.bn_params)
    output = tf.identity(output, name = 'conv_28_0_endp')
    ret_dict['conv_28_0_endp'] = output 
    # print("input d5 {}".format(output))                              
    output = tf.nn.max_pool(output, ksize = maxpool_ksz, strides = maxpool_strides, padding='SAME', data_format = self.data_format)
    current_conv_channel_sz *= 2 
    output = tc.layers.conv2d(output, current_conv_channel_sz, 3, 1, data_format = self.data_format,
                                    normalizer_fn=self.normalizer, normalizer_params=self.bn_params)
    output = tf.identity(output, name = 'conv_14_0_endp')
    ret_dict['conv_14_0_endp'] = output 
    
    # print("input d6 {}".format(output))                              
    output = tf.nn.max_pool(output, ksize = maxpool_ksz, strides = maxpool_strides, padding='SAME', data_format = self.data_format)
    current_conv_channel_sz *= 2 
    output = tc.layers.conv2d(output, current_conv_channel_sz, 3, 1, data_format = self.data_format,
                                    normalizer_fn=self.normalizer, normalizer_params=self.bn_params)
    output = tf.identity(output, name = 'conv_7_0_endp')
    ret_dict['last_conv'] = output 
    ret_dict['convfeats'] = ['conv_56_0_endp','conv_28_0_endp','conv_14_0_endp','last_conv'] 
    return ret_dict

def get_data(subset):
    # something that yields [[SHAPE, SHAPE, CHANNELS], [1]]
    ds = FakeData([[SHAPE, SHAPE, CHANNELS], [1]], 1000, random=False,
                  dtype=['float32', 'uint8'], domain=[(0, 255), (0, 10)])
    ds = MultiProcessRunnerZMQ(ds, 2)
    ds = BatchData(ds, BATCH_SIZE)
    return ds


# def get_config():
#     logger.auto_set_dir()

#     ds_train = get_data('train')
#     ds_test = get_data('test')

#     return TrainConfig(
#         model=Unet(),
#         data=QueueInput(ds_train),
#         callbacks=[
#             ModelSaver(),
#             InferenceRunner(ds_test, [ScalarStats('total_costs')]),
#         ],
#         steps_per_epoch=len(ds_train),
#         max_epoch=100,
#     )


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
#     parser.add_argument('--load', help='load model')
#     parser.add_argument('--config', help='config')
#     args = parser.parse_args()

#     if args.gpu:
#         os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
#     if args.config:
#         print(f"args.config : {args.config}")
#         DATA_BASEDIR = args.config
#     print(f"DATA_BASEDIR : {DATA_BASEDIR}")
#     register_coco(DATA_BASEDIR)

#     MODEL = Unet()

#     # config = get_config()
#     # config.session_init = SmartInit(args.load)
    
#     traincfg = TrainConfig(
#         model=MODEL,
#         dataflow=my_dataflow,
#         # data=my_inputsource, # alternatively, use an InputSource
#         callbacks=[...],    # some default callbacks are automatically applied
#         # some default monitors are automatically applied
#         steps_per_epoch=300,   # default to the size of your InputSource/DataFlow
#     )

#     launch_train_with_config(traincfg, SimpleTrainer())
