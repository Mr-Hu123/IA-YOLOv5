# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 16:48:11 2023

@author: HXF
"""
import tensorflow as tf
import core.utils_for_dip as ufd
from core.config import cfg
class DIP(object):
    def __init__(self, input_data, trainable, input_data_clean, defog_A=None, IcA=None):
        self.trainable        = trainable
        self.isp_flag = cfg.YOLO.ISP_FLAG
        self.recovery_loss, self.image_data = \
                            self._build_network(input_data, self.isp_flag, input_data_clean, defog_A, IcA)
        
    
    
    def _build_network(self, input_data, isp_flag, input_data_clean, defog_A, IcA):
        filtered_image_batch = input_data
        self.filter_params = input_data
        filter_imgs_series = []
        if isp_flag:
            with tf.variable_scope('extract_parameters_2'):
                input_data = tf.image.resize_images(input_data, [256, 256], method=tf.image.ResizeMethod.BILINEAR)
                filter_features = ufd.extract_parameters_2(input_data, cfg, self.trainable)

            # filter_features = tf.random_normal([1, 15], 0.5, 0.1)
            filters = cfg.filters
            filters = [x(filtered_image_batch, cfg) for x in filters]
            filter_parameters = []
            for j, filter in enumerate(filters):
                with tf.variable_scope('filter_%d' % j):
                    print('    creating filter:', j, 'name:', str(filter.__class__), 'abbr.',
                          filter.get_short_name())
                    print('      filter_features:', filter_features.shape)

                    filtered_image_batch, filter_parameter = filter.apply(
                        filtered_image_batch, filter_features, defog_A, IcA)
                    filter_parameters.append(filter_parameter)
                    filter_imgs_series.append(filtered_image_batch)
                    print('      output:', filtered_image_batch.shape)

            self.filter_params = filter_parameters
            # end_time = time.time()
            # print('filters所用时间：', end_time - start_time)
        # input_data_shape = tf.shape(input_data)
        # batch_size = input_data_shape[0]qwq
        recovery_loss = tf.reduce_sum(tf.pow(filtered_image_batch - input_data_clean, 2.0))#/(2.0 * batch_size)
        self.image_isped = filtered_image_batch
        self.filter_imgs_series = filter_imgs_series
        return recovery_loss, filtered_image_batch

    
    def get_recovery_loss(self):
        return self.recovery_loss
    
    def get_adjust_image(self):
        return self.image_data