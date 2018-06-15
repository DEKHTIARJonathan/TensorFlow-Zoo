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

from models import InceptionV4_Network

__all__ = [
    'InceptionV4_Test'
]


class InceptionV4_Test(CustomTestCase):

    @classmethod
    def setUpClass(cls):

        #######################################################################
        ####  =============    Placeholders Declaration      ============= ####
        #######################################################################

        input_plh = tf.placeholder(tf.float32, [None, 299, 299, 3], name='input_placeholder')

        #######################################################################
        ####  =============        Model Declaration         ============= ####
        #######################################################################
        inception_v4_net = InceptionV4_Network(include_FC_head=True, flatten_output=False)

        network = inception_v4_net(input_plh, reuse=False, is_train=False)
        _ = inception_v4_net.get_conv_layers(network)

        network_reuse = inception_v4_net(input_plh, reuse=True, is_train=False)
        cls.conv_shapes = inception_v4_net.get_conv_layers(network_reuse)

        cls.output_shape = network_reuse.outputs.shape

        network_reuse.print_layers()
        network_reuse.print_params(False)

        #######################################################################
        ####  =============        Run a sample Batch        ============= ####
        #######################################################################

        cls.max_prob = dict()
        cls.net_out_probs = dict()
        cls.most_likely_classID = dict()

        with tf.Session() as sess:
            inception_v4_net.load_pretrained(sess)

            cls.test_classes = [
                "koala",
                "panda",
                "weasel",
                "wombat"
            ]

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
                    img_resized = resize(img_raw, (299, 299), mode='reflect', anti_aliasing=False)  # Values in [0, 1]

                    if i == 0:
                        img_resized *= 255.0  # Values in [0, 255]
                    elif i == 1:
                        img_resized -= 0.5
                        img_resized *= 2.0  # Values in [-1, 1]

                    cls.net_out_probs["{}_{}".format(class_name, i)] = sess.run(
                        network_reuse.outputs,
                        feed_dict={input_plh: [img_resized]}
                    )[0]

                    cls.most_likely_classID["{}_{}".format(class_name, i)] = np.argmax(
                        cls.net_out_probs["{}_{}".format(class_name, i)]
                    )

                    cls.max_prob["{}_{}".format(class_name, i)] = cls.net_out_probs["{}_{}".format(class_name, i)][
                        cls.most_likely_classID["{}_{}".format(class_name, i)]
                    ]

    @classmethod
    def tearDownClass(cls):

        if tl.logging.get_verbosity() == tl.logging.DEBUG:
            print("\n\n###########################")

        for test_class in cls.test_classes:
            for i in range(3):

                if i == 0:
                    val_range = "[0, 255]"
                elif i == 1:
                    val_range = "[-1, 1]"
                else:
                    val_range = "[0, 1]"

                tl.logging.debug(
                    "InceptionV4 Network: [%s in %s] - Most Likely Class: %d with a probability of: %.5f - "
                    "Output Shape: %s" % (
                        test_class,
                        val_range,
                        cls.most_likely_classID["{}_{}".format(test_class, i)],
                        cls.max_prob["{}_{}".format(test_class, i)],
                        cls.net_out_probs["{}_{}".format(test_class, i)].shape
                    ))

        if tl.logging.get_verbosity() == tl.logging.DEBUG:
            print()

        for conv_shape in cls.conv_shapes:
            tl.logging.debug(
                "InceptionV4 Network: [%s] - Shape: %s" % (
                    conv_shape.name,
                    conv_shape.shape
                ))

        if tl.logging.get_verbosity() == tl.logging.DEBUG:
            print("###########################")

        tf.reset_default_graph()

    def test_most_likely_class(self):

        for i in range(3):
            self.assertEqual(self.most_likely_classID["koala_{}".format(i)], 106)
            self.assertEqual(self.most_likely_classID["panda_{}".format(i)], 389)
            self.assertEqual(self.most_likely_classID["weasel_{}".format(i)], 357)
            self.assertEqual(self.most_likely_classID["wombat_{}".format(i)], 107)

    def test_most_probable_result(self):

        for i in range(3):
            # Should output a confidence of 93.20%, allowing +/- a 2% error.
            self.assertTrue(abs((self.max_prob["koala_{}".format(i)] / 0.93) - 1) < 0.02)

            # Should output a confidence of 93.83%, allowing +/- a 2% error.
            self.assertTrue(abs((self.max_prob["panda_{}".format(i)] / 0.94) - 1) < 0.02)

            # Should output a confidence of 79.42%, allowing +/- a 2% error.
            self.assertTrue(abs((self.max_prob["weasel_{}".format(i)] / 0.79) - 1) < 0.02)

            # Should output a confidence of 89.22%, allowing +/- a 2% error.
            self.assertTrue(abs((self.max_prob["wombat_{}".format(i)] / 0.89) - 1) < 0.02)

    def test_output_shape(self):
        self.assertEqual(self.output_shape[1], 1001)

    def test_softmax_outputs(self):
        for class_name in self.test_classes:
            for i in range(3):
                self.assertTrue(np.min(self.net_out_probs["{}_{}".format(class_name, i)]) >= 0)
                self.assertTrue(np.max(self.net_out_probs["{}_{}".format(class_name, i)]) <= 1)

    def test_conv_shapes(self):
        self.assertEqual(self.conv_shapes[0].shape[1:], (149, 149, 32))
        self.assertEqual(self.conv_shapes[1].shape[1:], (147, 147, 32))
        self.assertEqual(self.conv_shapes[2].shape[1:], (147, 147, 64))
        self.assertEqual(self.conv_shapes[3].shape[1:], (73, 73, 160))
        self.assertEqual(self.conv_shapes[4].shape[1:], (71, 71, 192))
        self.assertEqual(self.conv_shapes[5].shape[1:], (35, 35, 384))
        self.assertEqual(self.conv_shapes[6].shape[1:], (35, 35, 384))
        self.assertEqual(self.conv_shapes[7].shape[1:], (35, 35, 384))
        self.assertEqual(self.conv_shapes[8].shape[1:], (35, 35, 384))
        self.assertEqual(self.conv_shapes[9].shape[1:], (35, 35, 384))
        self.assertEqual(self.conv_shapes[10].shape[1:], (17, 17, 1024))
        self.assertEqual(self.conv_shapes[11].shape[1:], (17, 17, 1024))
        self.assertEqual(self.conv_shapes[13].shape[1:], (17, 17, 1024))
        self.assertEqual(self.conv_shapes[14].shape[1:], (17, 17, 1024))
        self.assertEqual(self.conv_shapes[15].shape[1:], (17, 17, 1024))
        self.assertEqual(self.conv_shapes[16].shape[1:], (17, 17, 1024))
        self.assertEqual(self.conv_shapes[17].shape[1:], (17, 17, 1024))
        self.assertEqual(self.conv_shapes[18].shape[1:], (8, 8, 1536))
        self.assertEqual(self.conv_shapes[19].shape[1:], (8, 8, 1536))
        self.assertEqual(self.conv_shapes[20].shape[1:], (8, 8, 1536))
        self.assertEqual(self.conv_shapes[21].shape[1:], (8, 8, 1536))


if __name__ == '__main__':

    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

    tf.logging.set_verbosity(tf.logging.DEBUG)
    tl.logging.set_verbosity(tl.logging.DEBUG)

    unittest.main()
