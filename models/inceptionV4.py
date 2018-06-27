#! /usr/bin/python
# -*- coding: utf-8 -*-

import os

import tensorflow as tf
import tensorlayer as tl

from layers import conv_module
from layers import dense_module

from layers import block_inception_a
from layers import block_inception_b
from layers import block_inception_c

from layers import block_reduction_a
from layers import block_reduction_b

__all__ = [
    'InceptionV4_Network',
]


class InceptionV4_Network(object):
    """InceptionV4 model.
    """
    def __init__(
            self,
            include_FC_head  = True,
            flatten_output   = True
    ):

        self.include_FC_head  = include_FC_head
        self.flatten_output   = flatten_output

    def __call__(self, inputs, reuse=False, is_train=False):

        with tf.variable_scope("InceptionV4", reuse=reuse):

            preprocessed = inputs

            with tf.variable_scope("preprocessing"):

                max_val = tf.reduce_max(preprocessed)
                min_val = tf.reduce_min(preprocessed)

                need_int_rescale = tf.logical_and(
                    tf.greater(max_val, 1.0),
                    tf.greater_equal(min_val, 0.0)
                )

                need_float_rescale = tf.logical_and(
                    tf.less_equal(max_val, 1.0),
                    tf.greater_equal(min_val, 0.0)
                )

                preprocessed = tf.cond(
                    pred=need_int_rescale,
                    true_fn=lambda: tf.subtract(tf.divide(preprocessed, 127.5), 1.0),
                    false_fn=lambda: preprocessed
                )

                preprocessed = tf.cond(
                    pred=need_float_rescale,
                    true_fn=lambda: tf.multiply(tf.subtract(preprocessed, 0.5), 2.0),
                    false_fn=lambda: preprocessed
                )

            # Input Layers
            input_layer = tl.layers.InputLayer(preprocessed, name='input')

            # 299 x 299 x 3
            net, _ = conv_module(
                input_layer,
                n_out_channel=32,
                filter_size=(3, 3),
                strides=(2, 2),
                padding='VALID',
                batch_norm_init=None,
                is_train=is_train,
                use_batchnorm=True,
                activation_fn='ReLU',
                name='Conv2d_1a_3x3'
            )

            # 149 x 149 x 32
            net, _ = conv_module(
                net,
                n_out_channel=32,
                filter_size=(3, 3),
                strides=(1, 1),
                padding='VALID',
                batch_norm_init=None,
                is_train=is_train,
                use_batchnorm=True,
                activation_fn='ReLU',
                name='Conv2d_2a_3x3'
            )

            # 147 x 147 x 32
            net, _ = conv_module(
                net,
                n_out_channel=64,
                filter_size=(3, 3),
                strides=(1, 1),
                padding='SAME',
                batch_norm_init=None,
                is_train=is_train,
                use_batchnorm=True,
                activation_fn='ReLU',
                name='Conv2d_2b_3x3'
            )

            # 147 x 147 x 64
            with tf.variable_scope('Mixed_3a'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = tl.layers.MaxPool2d(net, (3, 3), strides=(2, 2), padding='VALID',
                                                   name='MaxPool_0a_3x3')

                with tf.variable_scope('Branch_1'):
                    branch_1, _ = conv_module(
                        net,
                        n_out_channel=96,
                        filter_size=(3, 3),
                        strides=(2, 2),
                        padding='VALID',
                        batch_norm_init=None,
                        is_train=is_train,
                        use_batchnorm=True,
                        activation_fn='ReLU',
                        name='Conv2d_0a_3x3'
                    )

                net = tl.layers.ConcatLayer([branch_0, branch_1], concat_dim=3)

            # 73 x 73 x 160
            with tf.variable_scope('Mixed_4a'):
                with tf.variable_scope('Branch_0'):
                    branch_0, _ = conv_module(
                        net,
                        n_out_channel=64,
                        filter_size=(1, 1),
                        strides=(1, 1),
                        padding='SAME',
                        batch_norm_init=None,
                        is_train=is_train,
                        use_batchnorm=True,
                        activation_fn='ReLU',
                        name='Conv2d_0a_1x1'
                    )

                    branch_0, _ = conv_module(
                        branch_0,
                        n_out_channel=96,
                        filter_size=(3, 3),
                        strides=(1, 1),
                        padding='VALID',
                        batch_norm_init=None,
                        is_train=is_train,
                        use_batchnorm=True,
                        activation_fn='ReLU',
                        name='Conv2d_1a_3x3'
                    )

                with tf.variable_scope('Branch_1'):
                    branch_1, _ = conv_module(
                        net,
                        n_out_channel=64,
                        filter_size=(1, 1),
                        strides=(1, 1),
                        padding='SAME',
                        batch_norm_init=None,
                        is_train=is_train,
                        use_batchnorm=True,
                        activation_fn='ReLU',
                        name='Conv2d_0a_1x1'
                    )

                    branch_1, _ = conv_module(
                        branch_1,
                        n_out_channel=64,
                        filter_size=(1, 7),
                        strides=(1, 1),
                        padding='SAME',
                        batch_norm_init=None,
                        is_train=is_train,
                        use_batchnorm=True,
                        activation_fn='ReLU',
                        name='Conv2d_0b_1x7'
                    )

                    branch_1, _ = conv_module(
                        branch_1,
                        n_out_channel=64,
                        filter_size=(7, 1),
                        strides=(1, 1),
                        padding='SAME',
                        batch_norm_init=None,
                        is_train=is_train,
                        use_batchnorm=True,
                        activation_fn='ReLU',
                        name='Conv2d_0c_7x1'
                    )

                    branch_1, _ = conv_module(
                        branch_1,
                        n_out_channel=96,
                        filter_size=(3, 3),
                        strides=(1, 1),
                        padding='VALID',
                        batch_norm_init=None,
                        is_train=is_train,
                        use_batchnorm=True,
                        activation_fn='ReLU',
                        name='Conv2d_1a_3x3'
                    )

                net = tl.layers.ConcatLayer([branch_0, branch_1], concat_dim=3)

            # 71 x 71 x 192
            with tf.variable_scope('Mixed_5a'):
                with tf.variable_scope('Branch_0'):
                    # 299 x 299 x 3
                    branch_0, _ = conv_module(
                        net,
                        n_out_channel=192,
                        filter_size=(3, 3),
                        strides=(2, 2),
                        padding='VALID',
                        batch_norm_init=None,
                        is_train=is_train,
                        use_batchnorm=True,
                        activation_fn='ReLU',
                        name='Conv2d_1a_3x3'
                    )

                with tf.variable_scope('Branch_1'):
                    branch_1 = tl.layers.MaxPool2d(net, (3, 3), strides=(2, 2), padding='VALID', name='MaxPool_1a_3x3')

                net = tl.layers.ConcatLayer([branch_0, branch_1], concat_dim=3)

            # 35 x 35 x 384
            # 4 x Inception-A blocks
            for idx in range(4):
                block_scope = 'Mixed_5' + chr(ord('b') + idx)
                net = block_inception_a(net, scope=block_scope, is_train=is_train)

            # 35 x 35 x 384
            # Reduction-A block
            net = block_reduction_a(net, scope='Mixed_6a', is_train=is_train)

            # 17 x 17 x 1024
            # 7 x Inception-B blocks
            for idx in range(7):
                block_scope = 'Mixed_6' + chr(ord('b') + idx)
                net = block_inception_b(net, scope=block_scope, is_train=is_train)

            # 17 x 17 x 1024
            # Reduction-B block
            net = block_reduction_b(net, scope='Mixed_7a', is_train=is_train)

            # 8 x 8 x 1536
            # 3 x Inception-C blocks
            for idx in range(3):
                block_scope = 'Mixed_7' + chr(ord('b') + idx)
                net = block_inception_c(net, scope=block_scope, is_train=is_train)

            if self.flatten_output and not self.include_FC_head:
               net = tl.layers.FlattenLayer(net, name='flatten')

            if self.include_FC_head:
                with tf.variable_scope("Logits", reuse=reuse):

                    # 8 x 8 x 1536
                    net = tl.layers.MeanPool2d(
                        net,
                        filter_size=net.outputs.get_shape()[1:3],
                        strides=(1, 1),
                        padding='VALID',
                        name='AvgPool_1a'
                    )

                    # 1 x 1 x 1536
                    net = tl.layers.DropoutLayer(net, keep=0.8, is_fix=True, is_train=is_train, name='Dropout_1b')
                    net = tl.layers.FlattenLayer(net, name='PreLogitsFlatten')

                    # 1536
                    net, _ = dense_module(
                        net,
                        n_units=1001,
                        activation_fn="softmax",
                        use_batchnorm=False,
                        batch_norm_init=None,
                        is_train=is_train,
                        name="Logits"
                    )

            if not reuse:
                self.network = net

            return net

    @staticmethod
    def get_conv_layers(network):

        conv_layers = [
            "Conv2d_1a_3x3/activation",
            "Conv2d_2a_3x3/activation",
            "Conv2d_2b_3x3/activation",
            "Mixed_3a/concat_layer",
            "Mixed_4a/concat_layer",
            "Mixed_5a/concat_layer",
            "Mixed_5b/concat_layer",
            "Mixed_5c/concat_layer",
            "Mixed_5d/concat_layer",
            "Mixed_5e/concat_layer",
            "Mixed_6a/concat_layer",
            "Mixed_6b/concat_layer",
            "Mixed_6c/concat_layer",
            "Mixed_6d/concat_layer",
            "Mixed_6e/concat_layer",
            "Mixed_6f/concat_layer",
            "Mixed_6g/concat_layer",
            "Mixed_6h/concat_layer",
            "Mixed_7a/concat_layer",
            "Mixed_7b/concat_layer",
            "Mixed_7c/concat_layer",
            "Mixed_7d/concat_layer"
        ]

        return [
            tl.layers.get_layers_with_name(
                network,
                name=layer_name,
                verbose=False
            )[0]
            for layer_name in conv_layers
        ]

    def load_pretrained(self, sess, ckpt_path='weights/inception_v4.ckpt'):

        tl.logging.info("Loading InceptionV4 Net Checkpoint ...")

        ckpt_path = os.path.join(os.path.realpath(__file__)[:-22], ckpt_path)

        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(
                "The file `%s` can not be found.\n"
                "Please download the checkpoint file at the following URL: %s" % (
                    ckpt_path,
                    'https://engineering-ai.com/models/inception_v4.ckpt'
                )
            )

        saver = tf.train.Saver(self.network.all_params)
        saver.restore(sess, ckpt_path)

        tl.logging.info("Finished loading InceptionV4 Net Checkpoint ...")
