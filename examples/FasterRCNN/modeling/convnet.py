import time
import tensorflow as tf
import numpy as np
#from models.layers import conv_layer, max_pool, fc_layer
import os
import cv2
import tensorflow.contrib as tc
from utils import log
from utils import common
import re

from datetime import timedelta

class Sconv7Net_v2(ConvNet):
    def __init__(self, input_shape, num_classes, gpulist, hparam_d, inputs=None, depth_multiplier=1.0, **kwargs):
        super(Sconv7Net_v2,self).__init__(input_shape, num_classes, gpulist, hparam_d, inputs, **kwargs)
        self.depth_multiplier = depth_multiplier

    def _build_model(self, inputx, hparam_d):
        log.debug("Sconv7Net_v2 _build_model")
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
        
        # print("input d7 {}".format(output))
        
        
        if hparam_d.get("model_type") == common.MlModelType.DET or hparam_d.get("model_type") == common.MlModelType.OCR:
            
            output = self._conv_ssd( output ,512, vc_suffix="conv_4_0_endp")
            output = tf.identity(output, name = 'conv_4_0_endp')
            ret_dict['conv_4_0_endp'] = output 
            output = self._conv_ssd( output ,256, vc_suffix="conv_2_0_endp")
            output = tf.identity(output, name = 'conv_2_0_endp')
            ret_dict['conv_2_0_endp'] = output 
            output = self._conv_ssd( output ,256, vc_suffix="conv_1_0_endp")
            output = tf.identity(output, name = 'conv_1_0_endp')
            ret_dict['conv_1_0_endp'] = output 
            output = self._conv_ssd( output ,256, vc_suffix="conv_e1_0_endp")
            output = tf.identity(output, name = 'conv_e1_0_endp')
            ret_dict['conv_e1_0_endp'] = output 
            #ret_dict['convfeats'] = ['conv_14_0_endp','last_conv','conv_4_0_endp', 'conv_2_0_endp', 'conv_1_0_endp','conv_e1_0_endp' ] 
            #ret_dict['convfeats'] = [ret_dict[convfeat_name] for convfeat_name in ret_dict['convfeats']]

            convfeat_name_list = [
                None, 
                None,
                'conv_56_0_endp', 
                'conv_28_0_endp',
                'conv_14_0_endp',
                'last_conv',
                'conv_4_0_endp',
                'conv_2_0_endp',
                'conv_1_0_endp',
                'conv_e1_0_endp'
            ]
            
            ret_dict['convfeats'] = self.select_pyramid_by_anchor_cfg([ret_dict.get(convfeat_name) for convfeat_name in convfeat_name_list])

            return ret_dict
        elif(hparam_d.get("model_type") == common.MlModelType.SEG) :
            ret_dict['convfeats'] = ['conv_56_0_endp','conv_28_0_endp','conv_14_0_endp','last_conv'] 
            return ret_dict

        with tf.variable_scope("logit_pred_hdr"): 
            #for CAM, variables_collections=["fclayer"] was added
            output = tc.layers.conv2d(output, self.num_classes, 1, activation_fn=None, data_format = self.data_format, normalizer_fn=None, normalizer_params=None,
            variables_collections=["fclayer"])
            ###########################
            # print("input d8 {}".format(output))
            avgpool_kernalsz = [output.shape[1], output.shape[2]]
            if self.data_format == 'NCHW':
               avgpool_kernalsz = [output.shape[2], output.shape[3]]
            output = tc.layers.avg_pool2d(output, avgpool_kernalsz, data_format = self.data_format)
            # print("input d9 {}".format(output))
            self._make_pred(output)
            ret_dict['logits'] = tf.reshape(output,[tf.shape(output)[0], output.shape[1]*output.shape[2]*output.shape[3]] )
            
            #self.d['argmaxed_pred'] = tf.argmax(self.d['pred'],axis=1,name="argmaxed_pred_")
        
        #CAM part
        with tf.name_scope("CAM"):
            ret_dict = self.cam_assign(ret_dict)
        ############################

        return ret_dict