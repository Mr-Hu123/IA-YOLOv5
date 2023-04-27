# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 16:54:04 2023

@author: HXF
"""
import tensorflow as tf
import tensorflow.contrib.layers as ly
import core.common as common
from util_filters import *

def extract_parameters(net, cfg, trainable):

    output_dim = cfg.num_filter_parameters
    # net = net - 0.5
    min_feature_map_size = 4
    print('extract_parameters CNN:')
    channels = cfg.base_channels

    print('    ', str(net.get_shape()))
    net = common.convolutional(net, filters_shape=(3, 3, 3, channels), trainable=trainable, name='ex_conv0',
                        downsample=True, activate=True, bn=False)
    net = common.convolutional(net, filters_shape=(3, 3, channels, 2*channels), trainable=trainable, name='ex_conv1',
                        downsample=True, activate=True, bn=False)
    net = common.convolutional(net, filters_shape=(3, 3, 2*channels, 2*channels), trainable=trainable, name='ex_conv2',
                        downsample=True, activate=True, bn=False)
    net = common.convolutional(net, filters_shape=(3, 3, 2*channels, 2*channels), trainable=trainable, name='ex_conv3',
                        downsample=True, activate=True, bn=False)
    net = common.convolutional(net, filters_shape=(3, 3, 2*channels, 2*channels), trainable=trainable, name='ex_conv4',
                        downsample=True, activate=True, bn=False)
    net = tf.reshape(net, [-1, 4096])
    features = ly.fully_connected(
        net,
        cfg.fc1_size,
        # cfg.fc1_size = 128
        scope='fc1',
        activation_fn=lrelu,
        weights_initializer=tf.contrib.layers.xavier_initializer())
    filter_features = ly.fully_connected(
        features,
        output_dim,
        scope='fc2',
        activation_fn=None,
        weights_initializer=tf.contrib.layers.xavier_initializer())
    return filter_features

def extract_parameters_2(net, cfg, trainable):
    output_dim = cfg.num_filter_parameters
    # output_dim = 15
    # net = net - 0.5
    

    min_feature_map_size = 4
    print('extract_parameters_2 CNN:')
    channels = 16
    print('    ', str(net.get_shape()))
    net = common.convolutional(net, filters_shape=(3, 3, 3, channels), trainable=trainable, name='ex_conv0',
                        downsample=True, activate=True, bn=False)

    net = common.convolutional(net, filters_shape=(3, 3, channels, 2*channels), trainable=trainable, name='ex_conv1',
                        downsample=True, activate=True, bn=False)

    net = common.convolutional(net, filters_shape=(3, 3, 2*channels, 2*channels), trainable=trainable, name='ex_conv2',
                        downsample=True, activate=True, bn=False)

    net = common.convolutional(net, filters_shape=(3, 3, 2*channels, 2*channels), trainable=trainable, name='ex_conv3',
                        downsample=True, activate=True, bn=False)

    net = common.convolutional(net, filters_shape=(3, 3, 2*channels, 2*channels), trainable=trainable, name='ex_conv4',
                        downsample=True, activate=True, bn=False)

    net = tf.reshape(net, [-1, 2048])
    
    features = ly.fully_connected(
        net,
        64,
        scope='fc1',
        activation_fn=lrelu,
        weights_initializer=tf.contrib.layers.xavier_initializer())
    filter_features = ly.fully_connected(
        features,
        output_dim,
        scope='fc2',
        activation_fn=None,
        weights_initializer=tf.contrib.layers.xavier_initializer())
    return filter_features