import os
import unittest

import numpy as np

import tensorflow as tf
import tensorlayer as tl

from tests.unittests_helper import CustomTestCase

from skimage.transform import resize
from imageio import imread

from models import VGG_Network

__all__ = [
    'VGG16_Test'
]


class VGG16_Test(CustomTestCase):

    def _set_up(self):

        #######################################################################
        ####  =============    Placeholders Declaration      ============= ####
        #######################################################################

        input_plh = tf.placeholder(tf.float32, [None, 224, 224, 3], name='input_placeholder')

        #######################################################################
        ####  =============        Model Declaration         ============= ####
        #######################################################################

        vgg_net = VGG_Network(include_FC_head=True)

        network = vgg_net(input_plh, reuse=False)
        _  = vgg_net.get_conv_layers(network)
        
        network_reuse = vgg_net(input_plh, reuse=True)
        _  = vgg_net.get_conv_layers(network_reuse)

        network_reuse.print_layers()
        network_reuse.print_params(False)

        #######################################################################
        ####  =============        Run a sample Batch        ============= ####
        #######################################################################

        with tf.Session() as sess:

            test_image_path = 'data/weasel.jpg'

            if os.getcwd().split(os.sep)[-1] != "tests":
                test_image_path = 'tests/' + test_image_path

            vgg_net.load_pretrained(sess)

            img_raw = imread(test_image_path)
            img_resized = resize(img_raw, (224, 224), mode='reflect', anti_aliasing=False) * 255

            self.net_out_probs = sess.run(network_reuse.outputs, feed_dict={input_plh: [img_resized]})[0]

            self.most_likely_classID = np.argmax(self.net_out_probs)
            self.max_prob = self.net_out_probs[self.most_likely_classID]

    def _tear_down(self):

        if tl.logging.get_verbosity() == tl.logging.DEBUG:
            print("\n\n###########################")

        tl.logging.debug("VGG Network: Most Likely Class: %d with a probability of: %.5f - Output Shape: %s" % (
            self.most_likely_classID,
            self.max_prob,
            self.net_out_probs.shape
        ))

        if tl.logging.get_verbosity() == tl.logging.DEBUG:
            print("###########################")

        tf.reset_default_graph()

    def test_most_likely_class(self):
        self.assertEqual(self.most_likely_classID, 356)

    def test_most_probable_result(self):
        # Should output a confidence of 89%, allowing +/- a 2% error.
        self.assertTrue(abs((self.max_prob / 0.89) - 1) < 0.02)

    def test_output_shape(self):
        self.assertEqual(self.net_out_probs.shape, (1000,))


if __name__ == '__main__':

    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

    tf.logging.set_verbosity(tf.logging.DEBUG)
    tl.logging.set_verbosity(tl.logging.DEBUG)

    unittest.main()


