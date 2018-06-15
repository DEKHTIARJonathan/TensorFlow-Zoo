import tensorflow as tf
import tensorlayer as tl

from . import activation_module

__all__ = [
    'deconv_module',
]


def deconv_module(
    prev_layer,
    n_out_channel,
    filter_size,
    output_size,
    padding,
    is_train=True,
    use_batchnorm=True,
    activation_fn=None,
    conv_init=tf.contrib.layers.xavier_initializer(uniform=True),
    deconv_init=tf.contrib.layers.xavier_initializer(uniform=True),
    batch_norm_init= tf.truncated_normal_initializer(mean=1., stddev=0.02),
    bias_init=tf.zeros_initializer(),
    deconv_type="with_upscale",
    name=None
):
    if activation_fn not in [
        "ReLU", "ReLU6", "Leaky_ReLU", "PReLU", "PReLU6",
        "PTReLU6", "CReLU", "ELU", "SELU", "tanh", "sigmoid",
        "softmax", None
    ]:
        raise Exception("Unknown 'activation_fn': %s" % activation_fn)

    if deconv_type not in ["normal", "with_upscale"]:
        raise Exception("Unknown 'deconv_type': %s" % deconv_type)

    conv_name       = 'conv2d' if name is None else name
    deconv_name     = 'deconv2d' if name is None else name
    bn_name         = 'batch_norm' if name is None else name + 'batch_norm'
    upsampling_name = 'upsampling' if name is None else name + 'upsampling'

    if deconv_type == "normal":
        layer = tl.layers.DeConv2d(
            layer       = prev_layer,
            n_filter    = n_out_channel,
            filter_size = filter_size,
            out_size    = output_size,
            strides     = (2, 2),
            padding     = padding,
            act         = None,
            W_init      = deconv_init,
            b_init      = None if use_batchnorm else bias_init,  # Not useful as the convolutions are batch normalized
            name        = deconv_name
        )
    else:

        layer = tl.layers.UpSampling2dLayer(
            prev_layer,
            size          = (2, 2),
            method        = 1, # ResizeMethod.NEAREST_NEIGHBOR
            align_corners = True,
            is_scale      = True,
            name          = upsampling_name
        )
        layer = tl.layers.Conv2d(
            layer,
            n_filter    = n_out_channel,
            filter_size = filter_size,
            strides     = (1, 1),
            padding     = padding,
            act         = None,
            W_init      = conv_init,
            b_init      = None if use_batchnorm else bias_init,  # Not useful as the convolutions are batch normalized
            name        = conv_name
        )

    if use_batchnorm:
        layer = tl.layers.BatchNormLayer(
            prev_layer    = layer,
            act           = None,
            is_train      = is_train,
            gamma_init    = batch_norm_init,
            name          = bn_name
        )

    logits = layer.outputs

    layer = activation_module(layer, activation_fn)

    return layer, logits
