#! /usr/bin/python
# -*- coding: utf-8 -*-

import os
import unittest

import numpy as np

import tensorflow as tf
import tensorlayer as tl

from tests.unittests_helper import CustomTestCase

from skimage.transform import resize
from imageio import imread

from models import SqueezeNet_Network

__all__ = [
    'SqueezeNet_Test'
]


class SqueezeNet_Test(CustomTestCase):

    def _set_up(self):

        #######################################################################
        ####  =============    Placeholders Declaration      ============= ####
        #######################################################################

        input_plh = tf.placeholder(tf.float32, (None, 224, 224, 3), name='input_placeholder')

        #######################################################################
        ####  =============        Model Declaration         ============= ####
        #######################################################################

        squeezenet = SqueezeNet_Network(include_convout_head=True, flatten_output=False)

        network = squeezenet(input_plh, reuse=False)
        _ = squeezenet.get_conv_layers(network)

        network_reuse = squeezenet(input_plh, reuse=True)
        self.conv_shapes = squeezenet.get_conv_layers(network_reuse)


        self.conv_shapes = squeezenet.get_conv_layers(network)

        self.output_shape = network_reuse.outputs.shape

        network_reuse.print_layers()
        network_reuse.print_params(False)

        #######################################################################
        ####  =============        Run a sample Batch        ============= ####
        #######################################################################

        self.max_prob = dict()
        self.net_out_probs = dict()
        self.most_likely_classID = dict()

        self.test_classes = list()

        with tf.Session() as sess:
            squeezenet.load_pretrained(sess)

            test_images = [
                ('koala', 'koala.jpg'),
                ('panda', 'panda.jpg'),
                ('weasel', 'weasel.jpg'),
                ('wombat', 'wombat.jpg')
            ]

            for class_name, test_image in test_images:

                self.test_classes.append(class_name)

                if os.getcwd().split(os.sep)[-1] != "tests":
                    test_image_path = 'tests/data/{}'.format(test_image)
                else:
                    test_image_path = 'data/{}'.format(test_image)

                img_raw = imread(test_image_path)

                for i in range(3):
                    img_resized = resize(img_raw, (224, 224), mode='reflect', anti_aliasing=False)  # Values in [0, 1]

                    if i == 0:
                        img_resized *= 255.0  # Values in [0, 255]

                    elif i == 1:
                        img_resized -= 0.5
                        img_resized *= 2.0  # Values in [-1, 1]

                    self.net_out_probs["{}_{}".format(class_name, i)] = sess.run(
                        network_reuse.outputs,
                        feed_dict={input_plh: [img_resized]}
                    )[0]

                    self.most_likely_classID["{}_{}".format(class_name, i)] = np.argmax(
                        self.net_out_probs["{}_{}".format(class_name, i)]
                    )

                    self.max_prob["{}_{}".format(class_name, i)] = self.net_out_probs["{}_{}".format(class_name, i)][
                        self.most_likely_classID["{}_{}".format(class_name, i)]
                    ]

    def _tear_down(self):

        if tl.logging.get_verbosity() == tl.logging.DEBUG:
            print("\n\n###########################")

        for test_class in self.test_classes:
            for i in range(3):

                if i == 0:
                    val_range = "[0, 255]"
                elif i == 1:
                    val_range = "[-1, 1]"
                else:
                    val_range = "[0, 1]"

                tl.logging.debug(
                    "SqueezeNet Network: [%s in %s] - Most Likely Class: %d with a probability of: %.5f - "
                    "Output Shape: %s" % (
                        test_class,
                        val_range,
                        self.most_likely_classID["{}_{}".format(test_class, i)],
                        self.max_prob["{}_{}".format(test_class, i)],
                        self.net_out_probs["{}_{}".format(test_class, i)].shape
                    ))

        if tl.logging.get_verbosity() == tl.logging.DEBUG:
            print()

        for conv_shape in self.conv_shapes:
            tl.logging.debug(
                "SqueezeNet Network: [%s] - Shape: %s" % (
                    conv_shape.name,
                    conv_shape.shape
                ))

        if tl.logging.get_verbosity() == tl.logging.DEBUG:
            print("###########################")

        tf.reset_default_graph()

    def test_most_likely_class(self):

        for i in range(3):
            self.assertEqual(self.most_likely_classID["koala_{}".format(i)], 105)
            self.assertEqual(self.most_likely_classID["panda_{}".format(i)], 388)
            self.assertEqual(self.most_likely_classID["weasel_{}".format(i)], 356)
            self.assertEqual(self.most_likely_classID["wombat_{}".format(i)], 106)

    def test_most_probable_result(self):

        for i in range(3):
            # Should output a confidence of 46.89%, allowing +/- a 2% error.
            self.assertTrue(abs((self.max_prob["koala_{}".format(i)] / 0.4689) - 1) < 0.02)

            # Should output a confidence of 96.69%, allowing +/- a 2% error.
            self.assertTrue(abs((self.max_prob["panda_{}".format(i)] / 0.9669) - 1) < 0.02)

            # Should output a confidence of 91.28%, allowing +/- a 2% error.
            self.assertTrue(abs((self.max_prob["weasel_{}".format(i)] / 0.9128) - 1) < 0.02)

            # Should output a confidence of 18.44%, allowing +/- a 2% error.
            self.assertTrue(abs((self.max_prob["wombat_{}".format(i)] / 0.1844) - 1) < 0.02)

    def test_output_shape(self):
        self.assertEqual(self.output_shape[1], 1000)

    def test_softmax_outputs(self):
        for class_name in self.test_classes:
            for i in range(3):
                self.assertTrue(np.min(self.net_out_probs["{}_{}".format(class_name, i)]) >= 0)
                self.assertTrue(np.max(self.net_out_probs["{}_{}".format(class_name, i)]) <= 1)

    def test_conv_shapes(self):
        self.assertEqual(self.conv_shapes[0].shape[1:], (55, 55, 128))
        self.assertEqual(self.conv_shapes[1].shape[1:], (55, 55, 128))
        self.assertEqual(self.conv_shapes[2].shape[1:], (27, 27, 256))
        self.assertEqual(self.conv_shapes[3].shape[1:], (27, 27, 256))
        self.assertEqual(self.conv_shapes[4].shape[1:], (13, 13, 384))
        self.assertEqual(self.conv_shapes[5].shape[1:], (13, 13, 384))
        self.assertEqual(self.conv_shapes[6].shape[1:], (13, 13, 512))
        self.assertEqual(self.conv_shapes[7].shape[1:], (13, 13, 512))
        self.assertEqual(self.conv_shapes[8].shape[1:], (13, 13, 1000))


if __name__ == '__main__':

    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

    tf.logging.set_verbosity(tf.logging.DEBUG)
    tl.logging.set_verbosity(tl.logging.DEBUG)

    unittest.main()
