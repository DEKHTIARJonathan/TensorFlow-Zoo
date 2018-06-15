#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorlayer as tl

__all__ = [
    'conv_block',
    'depthwise_conv_block'
]

def conv_block(block, n_filter, filter_size=(3, 3), strides=(1, 1), is_train=False, name='conv_block'):
    # ref: https://github.com/keras-team/keras/blob/master/keras/applications/mobilenet.py
    with tf.variable_scope(name):

        block = tl.layers.Conv2d(
            block,
            n_filter,
            filter_size,
            strides,
            padding='SAME',
            act=None,
            b_init=None,
            name='conv2d'
        )

        return tl.layers.BatchNormLayer(
            block,
            act=tf.nn.relu6,
            is_train=is_train,
            name='batchnorm'
        )

def depthwise_conv_block(block, n_filter, strides=(1, 1), is_train=False, name="depth_block"):
    with tf.variable_scope(name):

        block = tl.layers.DepthwiseConv2d(
            block,
            shape=(3, 3),
            strides=strides,
            padding='SAME',
            act=None,
            b_init=None,
            name='depthwise_conv2d'
        )

        block = tl.layers.BatchNormLayer(
            block,
            act=tf.nn.relu6,
            is_train=is_train,
            name='batchnorm1'
        )

        block = tl.layers.Conv2d(
            block,
            n_filter,
            filter_size=(1, 1),
            strides=(1, 1),
            padding='SAME',
            act=None,
            b_init=None,
            name='conv2d'
        )

        return tl.layers.BatchNormLayer(
            block,
            act=tf.nn.relu6,
            is_train=is_train,
            name='batchnorm2'
        )