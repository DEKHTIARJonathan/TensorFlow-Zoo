#! /usr/bin/python
# -*- coding: utf-8 -*-

import os

import numpy as np

import tensorflow as tf
import tensorlayer as tl

from layers import activation_module

__all__ = [
    'VGG_Network',
]


class VGG_Network(object):
    """VGG Network model.
    """
    def __init__(
            self,
            include_FC_head = True,
            flatten_output  = True
    ):

        self.include_FC_head = include_FC_head
        self.flatten_output  = flatten_output

        self.W_init = tf.contrib.layers.xavier_initializer(uniform=True)

    def __call__(self, inputs, reuse=False):

        with tf.variable_scope("VGG16", reuse=reuse):

            # Input Layers
            layer = tl.layers.InputLayer(inputs, name='input')

            # with tf.name_scope('preprocess') as scope:

            #     mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
            #     net_in.outputs = net_in.outputs - mean

            with tf.variable_scope("conv_layers"):

                with tf.variable_scope("h1"):

                    """ conv1 """
                    network = tl.layers.Conv2d(
                        layer,
                        n_filter=64,
                        filter_size=(3, 3),
                        strides=(1, 1),
                        act=tf.nn.relu,
                        padding='SAME',
                        name='conv1'
                    )

                    network = tl.layers.Conv2d(
                        network,
                        n_filter=64,
                        filter_size=(3, 3),
                        strides=(1, 1),
                        act=tf.nn.relu,
                        padding='SAME',
                        name='conv2'
                    )
                    network = tl.layers.MaxPool2d(
                        network,
                        filter_size=(2, 2),
                        strides=(2, 2),
                        padding='SAME',
                        name='pool'
                    )

                with tf.variable_scope("h2"):

                    """ conv2 """
                    network = tl.layers.Conv2d(
                        network,
                        n_filter=128,
                        filter_size=(3, 3),
                        strides=(1, 1),
                        act=tf.nn.relu,
                        padding='SAME',
                        name='conv1'
                    )

                    network = tl.layers.Conv2d(
                        network,
                        n_filter=128,
                        filter_size=(3, 3),
                        strides=(1, 1),
                        act=tf.nn.relu,
                        padding='SAME',
                        name='conv2'
                    )
                    network = tl.layers.MaxPool2d(
                        network,
                        filter_size=(2, 2),
                        strides=(2, 2),
                        padding='SAME',
                        name='pool'
                    )

                with tf.variable_scope("h3"):

                    """ conv3 """
                    network = tl.layers.Conv2d(
                        network,
                        n_filter=256,
                        filter_size=(3, 3),
                        strides=(1, 1),
                        act=tf.nn.relu,
                        padding='SAME',
                        name='conv1'
                    )

                    network = tl.layers.Conv2d(
                        network,
                        n_filter=256,
                        filter_size=(3, 3),
                        strides=(1, 1),
                        act=tf.nn.relu,
                        padding='SAME',
                        name='conv2'
                    )
                    network = tl.layers.Conv2d(
                        network,
                        n_filter=256,
                        filter_size=(3, 3),
                        strides=(1, 1),
                        act=tf.nn.relu,
                        padding='SAME',
                        name='conv3'
                    )
                    network = tl.layers.MaxPool2d(
                        network,
                        filter_size=(2, 2),
                        strides=(2, 2),
                        padding='SAME',
                        name='pool'
                    )

                with tf.variable_scope("h4"):

                    """ conv4 """
                    network = tl.layers.Conv2d(
                        network,
                        n_filter=512,
                        filter_size=(3, 3),
                        strides=(1, 1),
                        act=tf.nn.relu,
                        padding='SAME',
                        name='conv1'
                    )

                    network = tl.layers.Conv2d(
                        network,
                        n_filter=512,
                        filter_size=(3, 3),
                        strides=(1, 1),
                        act=tf.nn.relu,
                        padding='SAME',
                        name='conv2'
                    )
                    network = tl.layers.Conv2d(
                        network,
                        n_filter=512,
                        filter_size=(3, 3),
                        strides=(1, 1),
                        act=tf.nn.relu,
                        padding='SAME',
                        name='conv3'
                    )
                    network = tl.layers.MaxPool2d(
                        network,
                        filter_size=(2, 2),
                        strides=(2, 2),
                        padding='SAME',
                        name='pool'
                    )

                with tf.variable_scope("h5"):

                    """ conv5 """
                    network = tl.layers.Conv2d(
                        network,
                        n_filter=512,
                        filter_size=(3, 3),
                        strides=(1, 1),
                        act=tf.nn.relu,
                        padding='SAME',
                        name='conv1'
                    )

                    network = tl.layers.Conv2d(
                        network,
                        n_filter=512,
                        filter_size=(3, 3),
                        strides=(1, 1),
                        act=tf.nn.relu,
                        padding='SAME',
                        name='conv2'
                    )
                    network = tl.layers.Conv2d(
                        network,
                        n_filter=512,
                        filter_size=(3, 3),
                        strides=(1, 1),
                        act=tf.nn.relu,
                        padding='SAME',
                        name='conv3'
                    )
                    network = tl.layers.MaxPool2d(
                        network,
                        filter_size=(2, 2),
                        strides=(2, 2),
                        padding='SAME',
                        name='pool'
                    )

            if self.flatten_output or self.include_FC_head:
                with tf.variable_scope("conv_2_flat"):
                    network = tl.layers.FlattenLayer(network, name='flatten')

            if self.include_FC_head:
                with tf.variable_scope("dense_layers"):

                    network = tl.layers.DenseLayer(network, n_units=4096, act=tf.nn.relu, name='fc1')
                    network = tl.layers.DenseLayer(network, n_units=4096, act=tf.nn.relu, name='fc2')
                    logits  = tl.layers.DenseLayer(network, n_units=1000, act=None, name='fc3')

                    network = activation_module(logits, "softmax")

            if not reuse:
                self.network = network

            return network

    @staticmethod
    def get_conv_layers(network):

        conv_layers = [
            "conv_layers/h1/pool/MaxPool",
            "conv_layers/h2/pool/MaxPool",
            "conv_layers/h3/pool/MaxPool",
            "conv_layers/h4/pool/MaxPool",
            "conv_layers/h5/pool/MaxPool",
        ]

        return [
            tl.layers.get_layers_with_name(
                network,
                name=layer_name,
                verbose=False
            )[0]
            for layer_name in conv_layers
        ]

    def load_pretrained_from_weights(self, sess, weights_path='../weights/vgg16_weights.npz'):

        tl.logging.info("Loading VGG16 Net weights ...")

        weights_path = os.path.join(os.path.realpath(__file__)[:-16], weights_path)

        if not os.path.isfile(weights_path):
            raise FileNotFoundError("The file `%s` can not be found." % weights_path)

        n_params = len(self.network.all_params)

        npz = np.load(weights_path)
        params = list()

        for i, val in enumerate(sorted(npz.items())):
            params.append(val[1])

            if i >= n_params - 1:
                break

        tl.files.assign_params(sess, params, self.network)

        tl.logging.info("Finished loading VGG16 Net weights ...")

    def load_pretrained(self, sess, ckpt_path='weights/vgg16.ckpt'):

        tl.logging.info("Loading VGG16 Checkpoint ...")

        ckpt_path = os.path.join(os.path.realpath(__file__)[:-16], ckpt_path)

        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(
                "The file `%s` can not be found.\n"
                "Please download the checkpoint file at the following URL: %s" % (
                    ckpt_path,
                    'http://www.smarter-engineering.com/models/vgg16.ckpt'
                )
            )

        saver = tf.train.Saver(self.network.all_params)
        saver.restore(sess, ckpt_path)

        tl.logging.info("Finished loading VGG16 Checkpoint ...")
