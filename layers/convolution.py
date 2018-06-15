import tensorflow as tf
import tensorlayer as tl

from . import activation_module

__all__ = [
    'conv_module',
]


def conv_module(
    prev_layer,
    n_out_channel,
    filter_size,
    strides,
    padding,
    is_train=True,
    use_batchnorm=True,
    activation_fn=None,
    conv_init=tf.contrib.layers.xavier_initializer(uniform=True),
    batch_norm_init= tf.truncated_normal_initializer(mean=1., stddev=0.02),
    bias_init=tf.zeros_initializer(),
    name=None
):

    if activation_fn not in [
        "ReLU", "ReLU6", "Leaky_ReLU", "PReLU", "PReLU6",
        "PTReLU6", "CReLU", "ELU", "SELU", "tanh", "sigmoid",
        "softmax", None
    ]:
        raise Exception("Unknown 'activation_fn': %s" % activation_fn)

    conv_name = 'conv2d' if name is None else name
    bn_name   = 'batch_norm' if name is None else name + '/BatchNorm'

    layer = tl.layers.Conv2d(
        prev_layer,
        n_filter    = n_out_channel,
        filter_size = filter_size,
        strides     = strides,
        padding     = padding,
        act         = None,
        W_init      = conv_init,
        b_init      = None if use_batchnorm else bias_init,  # Not useful as the convolutions are batch normalized
        name        = conv_name
    )

    if use_batchnorm:

        layer = tl.layers.BatchNormLayer(
            layer,
            act        = None,
            is_train   = is_train,
            gamma_init = batch_norm_init,
            name       = bn_name
        )

    logits = layer.outputs

    layer = activation_module(layer, activation_fn, name=conv_name)

    return layer, logits
