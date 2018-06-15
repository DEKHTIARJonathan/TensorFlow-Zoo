#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorlayer as tl

from . import activation_module

__all__ = [
    'dense_module',
]


def dense_module(
    prev_layer,
    n_units,
    is_train,
    use_batchnorm=True,
    activation_fn=None,
    dense_init=tf.contrib.layers.xavier_initializer(uniform=True),
    batch_norm_init=tf.truncated_normal_initializer(mean=1., stddev=0.02),
    bias_init=tf.zeros_initializer(),
    name=None
):

    if activation_fn not in [
        "ReLU", "ReLU6", "Leaky_ReLU", "PReLU", "PReLU6",
        "PTReLU6", "CReLU", "ELU", "SELU", "tanh", "sigmoid",
        "softmax", None
    ]:
        raise Exception("Unknown 'activation_fn': %s" % activation_fn)

    # Flatten: Conv to FC
    if prev_layer.outputs.get_shape().__len__() != 2:  # The input dimension must be rank 2
        layer = tl.layers.FlattenLayer(prev_layer, name='flatten')

    else:
        layer = prev_layer

    layer = tl.layers.DenseLayer(
        layer,
        n_units = n_units,
        act     = None,
        W_init  = dense_init,
        b_init  = None if use_batchnorm else bias_init,  # Not useful as the convolutions are batch normalized
        name    = 'dense' if name is None else name
    )

    if use_batchnorm:
        layer = tl.layers.BatchNormLayer(
            layer,
            act        = None,
            is_train   = is_train,
            gamma_init = batch_norm_init,
            name       = 'batch_norm'
        )

    logits = layer.outputs

    layer = activation_module(layer, activation_fn)

    return layer, logits
