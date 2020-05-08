# -*- coding: utf-8 -*-
# Author: Your Name <your@email.com>

import argparse
import os
import tensorflow as tf

from tensorpack import *

"""
This is a boiler-plate template.
All code is in this file is the most minimalistic way to solve a deep-learning problem with cross-validation.
"""

BATCH_SIZE = 4
SHAPE = 512
CHANNELS = 3


class Model(ModelDesc):
    def inputs(self):
        return [tf.TensorSpec((None, SHAPE, SHAPE, CHANNELS), tf.float32, 'input1'),
                tf.TensorSpec((None,), tf.int32, 'input2')]

    def build_graph(self, input1, input2):

        cost = tf.identity(input1 - input2, name='total_costs')
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


def get_config():
    logger.auto_set_dir()

    ds_train = get_data('train')
    ds_test = get_data('test')

    return TrainConfig(
        model=Model(),
        data=QueueInput(ds_train),
        callbacks=[
            ModelSaver(),
            InferenceRunner(ds_test, [ScalarStats('total_costs')]),
        ],
        steps_per_epoch=len(ds_train),
        max_epoch=100,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='load model')
    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    config = get_config()
    config.session_init = SmartInit(args.load)

    launch_train_with_config(config, SimpleTrainer())
