import tensorflow as tf
import tensorlayer as tl

__all__ = [
    'activation_module',
]


def activation_module(layer, activation_fn, leaky_relu_alpha=0.2, name=None):

    act_name = name + "/activation" if name is not None else "activation"

    if activation_fn not in [
        "ReLU", "ReLU6", "Leaky_ReLU", "PReLU", "PReLU6",
        "PTReLU6", "CReLU", "ELU", "SELU", "tanh", "sigmoid",
        "softmax", None
    ]:
        raise Exception("Unknown 'activation_fn': %s" % activation_fn)

    elif activation_fn == "ReLU":
        layer = tl.layers.LambdaLayer(
            prev_layer=layer,
            fn=tf.nn.relu,
            name=act_name
        )

    elif activation_fn == "ReLU6":
        layer = tl.layers.LambdaLayer(
            prev_layer=layer,
            fn=tf.nn.relu6,
            name=act_name
        )

    elif activation_fn == "Leaky_ReLU":
        layer = tl.layers.LambdaLayer(
            prev_layer=layer,
            fn=tf.nn.leaky_relu,
            fn_args={'alpha': leaky_relu_alpha},
            name=act_name
        )

    elif activation_fn == "PReLU":
        layer = tl.layers.PReluLayer(
            prev_layer     = layer,
            channel_shared = False,
            name           = act_name
        )

    elif activation_fn == "PReLU6":
        layer = tl.layers.PRelu6Layer(
            prev_layer     = layer,
            channel_shared = False,
            name           = act_name
        )

    elif activation_fn == "PTReLU6":
        layer = tl.layers.PTRelu6Layer(
            prev_layer     = layer,
            channel_shared = False,
            name           = act_name
        )

    elif activation_fn == "CReLU":
        layer = tl.layers.LambdaLayer(
            prev_layer=layer,
            fn=tf.nn.crelu,
            name=act_name
        )

    elif activation_fn == "ELU":
        layer = tl.layers.LambdaLayer(
            prev_layer=layer,
            fn=tf.nn.elu,
            name=act_name
        )

    elif activation_fn == "SELU":
        layer = tl.layers.LambdaLayer(
            prev_layer=layer,
            fn=tf.nn.selu,
            name=act_name
        )

    elif activation_fn == "tanh":
        layer = tl.layers.LambdaLayer(
            prev_layer=layer,
            fn=tf.nn.tanh,
            name=act_name
        )

    elif activation_fn == "sigmoid":
        layer = tl.layers.LambdaLayer(
            prev_layer=layer,
            fn=tf.nn.sigmoid,
            name=act_name
        )

    elif activation_fn == "softmax":
        layer = tl.layers.LambdaLayer(
            prev_layer=layer,
            fn=tf.nn.softmax,
            name=act_name
        )

    return layer
