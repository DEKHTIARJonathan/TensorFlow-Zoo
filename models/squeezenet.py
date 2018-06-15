#! /usr/bin/python
# -*- coding: utf-8 -*-

import os

import tensorflow as tf
import tensorlayer as tl

from layers import activation_module
from layers import fire_module

__all__ = [
    'SqueezeNet_Network',
]


class SqueezeNet_Network(object):
    """SqueezeNet Network model.
    """
    def __init__(
            self,
            include_convout_head = True,
            flatten_output       = True
    ):

        self.include_convout_head = include_convout_head
        self.flatten_output       = flatten_output

    def __call__(self, inputs, reuse=False):

        with tf.variable_scope("squeezenet", reuse=reuse):

            preprocessed = inputs

            with tf.variable_scope("preprocessing"):
                max_val = tf.reduce_max(preprocessed)
                min_val = tf.reduce_min(preprocessed)

                need_0_1_rescale = tf.logical_and(  # values in [0, 1]
                    tf.less_equal(max_val, 1.0),
                    tf.greater_equal(min_val, 0.0)
                )

                need_min1_1_rescale = tf.logical_and(
                    tf.less_equal(max_val, 1.0),
                    tf.less(min_val, 0.0)
                )

                preprocessed = tf.cond(
                    pred=need_0_1_rescale,
                    true_fn=lambda: tf.multiply(preprocessed, 255),
                    false_fn=lambda: preprocessed
                )

                preprocessed = tf.cond(
                    pred=need_min1_1_rescale,
                    true_fn=lambda: tf.multiply(tf.add(preprocessed, 1.0), 127.5),
                    false_fn=lambda: preprocessed
                )

            with tf.variable_scope("input"):
                network = tl.layers.InputLayer(preprocessed)

            network = tl.layers.Conv2d(network, 64, (3, 3), (2, 2), tf.nn.relu, padding='SAME', name='conv1')

            network = tl.layers.MaxPool2d(network, (3, 3), (2, 2), 'VALID', name='maxpool1')

            # Name: squeezenet/fire2/concat
            network = fire_module(network, 16, 64, name="fire2")

            # Name: squeezenet/fire3/concat
            network = fire_module(network, 16, 64, name="fire3")

            network = tl.layers.MaxPool2d(network, (3, 3), (2, 2), 'VALID', name='maxpool3')

            # Name: squeezenet/fire4/concat
            network = fire_module(network, 32, 128, name="fire4")

            # Name: squeezenet/fire5/concat
            network = fire_module(network, 32, 128, name="fire5")

            network = tl.layers.MaxPool2d(network, (3, 3), (2, 2), 'VALID', name='maxpool5')

            # Name: squeezenet/fire6/concat
            network = fire_module(network, 48, 192, name="fire6")

            # Name: squeezenet/fire7/concat
            network = fire_module(network, 48, 192, name="fire7")

            # Name: squeezenet/fire8/concat
            network = fire_module(network, 64, 256, name="fire8")

            # Name: squeezenet/fire9/concat
            network = fire_module(network, 64, 256, name="fire9")

            with tf.variable_scope("output"):

                # Name: squeezenet/output/conv10
                network = tl.layers.Conv2d(network, 1000, (1, 1), (1, 1), tf.nn.relu, padding='SAME', name='conv10')

                if self.flatten_output and not self.include_convout_head:
                    network = tl.layers.FlattenLayer(network, name='flatten')

                elif self.include_convout_head:

                    logits = tl.layers.GlobalMeanPool2d(network, name='avgpool10')

                    network = activation_module(logits, "softmax")

            if not reuse:
                self.network = network

            return network

    @staticmethod
    def get_conv_layers(network):

        conv_layers = [
            "fire2/concat",  # Shape: [None, 55, 55, 128]
            "fire3/concat",  # Shape: [None, 55, 55, 128]
            "fire4/concat",  # Shape: [None, 27, 27, 256]
            "fire5/concat",  # Shape: [None, 27, 27, 256]
            "fire6/concat",  # Shape: [None, 13, 13, 384]
            "fire7/concat",  # Shape: [None, 13, 13, 384]
            "fire8/concat",  # Shape: [None, 13, 13, 512]
            "fire9/concat",  # Shape: [None, 13, 13, 512]
            "output/conv10/Relu"  # Shape: [None, 13, 13, 1000]
        ]

        return [
            tl.layers.get_layers_with_name(
                network,
                name=layer_name,
                verbose=False
            )[0]
            for layer_name in conv_layers
        ]

    def load_pretrained_from_weights(self, sess, weights_path='weights/squeezenet.npz'):

        tl.logging.info("Loading SqueezeNet weights ...")

        weights_path = os.path.join(os.path.realpath(__file__)[:-21], weights_path)

        if not os.path.isfile(weights_path):
            raise FileNotFoundError("The file `%s` can not be found." % weights_path)

        try:

            if tl.files.file_exists(weights_path):
                tl.files.load_and_assign_npz(sess=sess, name=weights_path, network=self.network)

            else:
                raise Exception(
                    "please download the pre-trained squeezenet.npz from https://github.com/tensorlayer/pretrained-models"
                )

        except AttributeError:
            raise RuntimeError("The %s model has not been created yet" % self.__class__.__name__)

        tl.logging.info("Finished loading SqueezeNet Net weights ...")

    def load_pretrained(self, sess, ckpt_path='weights/squeezenet.ckpt'):

        tl.logging.info("Loading SqueezeNet Checkpoint ...")

        ckpt_path = os.path.join(os.path.realpath(__file__)[:-21], ckpt_path)

        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(
                "The file `%s` can not be found.\n"
                "Please download the checkpoint file at the following URL: %s" % (
                    ckpt_path,
                    'http://www.smarter-engineering.com/models/squeezenet.ckpt'
                )
            )

        saver = tf.train.Saver(self.network.all_params)
        saver.restore(sess, ckpt_path)

        tl.logging.info("Finished loading SqueezeNet Checkpoint ...")
