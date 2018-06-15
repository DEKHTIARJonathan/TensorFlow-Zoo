#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorlayer as tl

__all__ = [
    'fire_module',
]


def fire_module(inputs, squeeze_depth, expand_depth, name):
    """Fire module: squeeze input filters, then apply spatial convolutions."""

    with tf.variable_scope(name, "fire", [inputs]):

        squeezed = tl.layers.Conv2d(
            inputs,
            squeeze_depth,
            (1, 1),
            (1, 1),
            tf.nn.relu,
            'SAME',
            name='squeeze'
        )

        e1x1 = tl.layers.Conv2d(
            squeezed,
            expand_depth,
            (1, 1),
            (1, 1),
            tf.nn.relu,
            'SAME',
            name='e1x1'
        )

        e3x3 = tl.layers.Conv2d(
            squeezed,
            expand_depth,
            (3, 3),
            (1, 1),
            tf.nn.relu,
            'SAME',
            name='e3x3'
        )

        return tl.layers.ConcatLayer([e1x1, e3x3], concat_dim=3, name='concat')