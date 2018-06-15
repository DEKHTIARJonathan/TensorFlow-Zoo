import os

import tensorflow as tf
import tensorlayer as tl

from layers import activation_module

from layers import conv_block
from layers import depthwise_conv_block

__all__ = [
    'MobileNet_Network',
]


class MobileNet_Network(object):
    """MobileNet V2 Network model.
    Architecture: https://arxiv.org/abs/1801.04381
    The base model gives 72.2% accuracy on ImageNet, with 300MMadds,
    3.4 M parameters.
    """
    def __init__(
            self,
            include_convout_head = True,
            flatten_output       = True
    ):

        self.include_convout_head = include_convout_head
        self.flatten_output       = flatten_output

    def __call__(self, inputs, is_train=False, reuse=False):

        with tf.variable_scope("mobilenet_v2", reuse=reuse):

            preprocessed = inputs

            with tf.variable_scope("preprocessing"):
                max_val = tf.reduce_max(preprocessed)
                min_val = tf.reduce_min(preprocessed)

                need_0_255_rescale = tf.logical_and(  # values in [0, 255]
                    tf.greater(max_val, 1.0),
                    tf.greater_equal(min_val, 0.0)
                )

                need_min1_1_rescale = tf.logical_and(
                    tf.less_equal(max_val, 1.0),
                    tf.less(min_val, 0.0)
                )

                preprocessed = tf.cond(
                    pred=need_0_255_rescale,
                    true_fn=lambda: tf.divide(preprocessed, 255.0),
                    false_fn=lambda: preprocessed
                )

                preprocessed = tf.cond(
                    pred=need_min1_1_rescale,
                    true_fn=lambda: tf.add(tf.divide(preprocessed, 2.0), 0.5),
                    false_fn=lambda: preprocessed
                )

            with tf.variable_scope("input"):
                network = tl.layers.InputLayer(preprocessed)

            network = conv_block(network, 32, strides=(2, 2), is_train=is_train, name="conv")

            network = depthwise_conv_block(network, 64, is_train=is_train, name="depth1")

            network = depthwise_conv_block(network, 128, strides=(2, 2), is_train=is_train, name="depth2")
            network = depthwise_conv_block(network, 128, is_train=is_train, name="depth3")

            network = depthwise_conv_block(network, 256, strides=(2, 2), is_train=is_train, name="depth4")
            network = depthwise_conv_block(network, 256, is_train=is_train, name="depth5")

            network = depthwise_conv_block(network, 512, strides=(2, 2), is_train=is_train, name="depth6")
            network = depthwise_conv_block(network, 512, is_train=is_train, name="depth7")
            network = depthwise_conv_block(network, 512, is_train=is_train, name="depth8")
            network = depthwise_conv_block(network, 512, is_train=is_train, name="depth9")
            network = depthwise_conv_block(network, 512, is_train=is_train, name="depth10")
            network = depthwise_conv_block(network, 512, is_train=is_train, name="depth11")

            network = depthwise_conv_block(network, 1024, strides=(2, 2), is_train=is_train, name="depth12")
            network = depthwise_conv_block(network, 1024, is_train=is_train, name="depth13")

            with tf.variable_scope("output"):

                if self.flatten_output and not self.include_convout_head:
                    network = tl.layers.FlattenLayer(network, name='flatten_output')

                elif self.include_convout_head:
                    network = tl.layers.GlobalMeanPool2d(network, name="global_avgpool_2d")
                    network = tl.layers.ReshapeLayer(network, shape=(-1, 1, 1, 1024), name="reshape")

                    network = tl.layers.Conv2d(
                        network,
                        1000,
                        filter_size=(1, 1),
                        strides=(1, 1),
                        padding='VALID',
                        act=None,
                        name='conv_out'
                    )

                    logits = tl.layers.FlattenLayer(network, name="flatten_output")

                    network = activation_module(logits, "softmax")

            if not reuse:
                self.network = network

            return network

    @staticmethod
    def get_conv_layers(network):

        conv_layers = [
            'conv/batchnorm',      # [None, 112, 112,   32]
            'depth1/batchnorm2',   # [None, 112, 112,   64]
            'depth2/batchnorm2',   # [None,  56,  56,  128]
            'depth3/batchnorm2',   # [None,  56,  56,  128]
            'depth4/batchnorm2',   # [None,  28,  28,  256]
            'depth5/batchnorm2',   # [None,  28,  28,  256]
            'depth6/batchnorm2',   # [None,  14,  14,  512]
            'depth7/batchnorm2',   # [None,  14,  14,  512]
            'depth8/batchnorm2',   # [None,  14,  14,  512]
            'depth9/batchnorm2',   # [None,  14,  14,  512]
            'depth10/batchnorm2',  # [None,  14,  14,  512]
            'depth11/batchnorm2',  # [None,  14,  14,  512]
            'depth12/batchnorm2',  # [None,   7,   7, 1024]
            'depth13/batchnorm2',  # [None,   7,   7, 1024]
        ]

        return [
            tl.layers.get_layers_with_name(
                network,
                name=layer_name,
                verbose=False
            )[0]
            for layer_name in conv_layers
        ]

    def load_pretrained_from_weights(self, sess, weights_path='weights/mobilenet.npz'):

        tl.logging.info("Loading MobileNet weights ...")

        weights_path = os.path.join(os.path.realpath(__file__)[:-20], weights_path)

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

        tl.logging.info("Finished loading MobileNet Net weights ...")

    def load_pretrained(self, sess, ckpt_path='weights/mobilenet.ckpt'):

        tl.logging.info("Loading MobileNet Checkpoint ...")

        ckpt_path = os.path.join(os.path.realpath(__file__)[:-20], ckpt_path)

        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError("The file `%s` can not be found." % ckpt_path)

        saver = tf.train.Saver(self.network.all_params)
        saver.restore(sess, ckpt_path)

        tl.logging.info("Finished loading MobileNet Checkpoint ...")


if __name__ == "__main__":

    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)

    from skimage.transform import resize
    from imageio import imread

    import numpy as np

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

    tf.logging.set_verbosity(tf.logging.DEBUG)
    tl.logging.set_verbosity(tl.logging.DEBUG)

    input_plh = tf.placeholder(tf.float32, (None, 224, 224, 3), name='input_placeholder')

    #######################################################################
    ####  =============        Model Declaration         ============= ####
    #######################################################################

    mobilenet = MobileNet_Network(include_convout_head=True, flatten_output=False)

    network = mobilenet(input_plh, is_train=False, reuse=False)
    logits  = MobileNet_Network.get_conv_layers(network)

    network2 = mobilenet(input_plh, is_train=False, reuse=True)
    logits2  = MobileNet_Network.get_conv_layers(network2)

    network.print_layers()
    network.print_params(False)

    with tf.Session() as sess:
        mobilenet.load_pretrained(sess)

        test_images = [
            ('koala', 'koala.jpg'),
            ('panda', 'panda.jpg'),
            ('weasel', 'weasel.jpg'),
            ('wombat', 'wombat.jpg')
        ]

        for class_name, test_image in test_images:

            if os.getcwd().split(os.sep)[-1] != "tests":
                test_image_path = 'tests/data/{}'.format(test_image)
            else:
                test_image_path = 'data/{}'.format(test_image)

            img_raw = imread(test_image_path)

            for i in range(3):
                img_resized = resize(img_raw, (224, 224), mode='reflect', anti_aliasing=False)  # Values in [0, 1]

                if i == 0:
                    img_resized *= 255.0  # Values in [0, 1]

                elif i == 1:
                    img_resized -= 0.5
                    img_resized *= 2.0  # Values in [-1, 1]

                probs = sess.run(network2.outputs, feed_dict={input_plh: [img_resized]})[0]

                print("Mode: %d - Class: %s - Prob: %f" % (i, np.argmax(probs), probs[np.argmax(probs)]))