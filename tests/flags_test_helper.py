import sys
import tensorflow as tf

__all__ = [
    'get_test_flags'
]


def get_test_flags():

    if tf.VERSION < '1.5':
        import argparse
        tf.app.flags.FLAGS = tf.flags._FlagValues()
        tf.app.flags._global_parser = argparse.ArgumentParser()

    else:
        for name in list(tf.app.flags.FLAGS):
            delattr(tf.app.flags.FLAGS, name)

    flags = tf.app.flags

    # training params
    flags.DEFINE_integer("epoch", 1, "Number of epochs to train. [25]")
    flags.DEFINE_integer("batch_size", 2, "Number of images in batch [64]")
    flags.DEFINE_integer("sample_size", 32, "Number of images to sample [64]")
    flags.DEFINE_integer("sample_step", 50, "The interval of generating sample. [500]")

    # model parameters
    flags.DEFINE_integer("image_size", 256, "The size in pixel of the images [64]")
    flags.DEFINE_integer("c_dim", 1, "Input Color Channel Dimension [3: RGB | 1: Grayscale]")
    flags.DEFINE_integer("latent_size", 100, "The size of the output vector z in the latent space [100]")
    flags.DEFINE_integer("first_conv_depth", 64, "Number of filters for the first convolution of the encoder [64]")
    flags.DEFINE_integer("last_conv_size_discriminator", 16, "Number of filters for the last convolution in the discriminator [64]")
    flags.DEFINE_integer("max_conv_filters_discriminator", 256, "Number of filters for the last convolution in the encoder [64]")
    flags.DEFINE_integer("last_conv_size_encoder", 8, "Number of filters for the last convolution in the encoder [64]")
    flags.DEFINE_integer("max_conv_filters_encoder", 256, "Number of filters for the last convolution in the encoder [64]")
    flags.DEFINE_string("deconv_type", "with_upscale", "Type of Deconvolution Used [normal or with_upscale]")
    flags.DEFINE_string("encoder_activation", "PReLU",  "Activation Function to be used in the encoder for the conv layers ['PReLU']")
    flags.DEFINE_string("generator_activation", "PReLU",  "Activation Function to be used in the generator for the deconv layers ['PReLU']")
    flags.DEFINE_string("discriminator_activation", "PReLU",  "Activation Function to be used in the discriminator for the conv layers ['PReLU']")

    # loss parameters
    flags.DEFINE_boolean("use_WGAN_loss", False,  "The GAN implementation follows the WGAN implementation [True]")
    flags.DEFINE_float("WGAN_d_weight_clip", 0.01, "The value 'c' used to clipped the weights between [-c, c]")
    flags.DEFINE_float("lambda_val_res", 0.1, "Strength of the discrimination loss at evaluation time")
    flags.DEFINE_float("lambda_val_per", 0.1, "Strength of the perceptual loss at evaluation time")
    flags.DEFINE_float("label_smoothing", 0.1, "Value use to apply label smoothing in the discriminator loss function [0.1]")

    # optimizer settings
    flags.DEFINE_float("learning_rate_e", 2e-4, "Learning rate for the Encoder optimizer [2e-4]")
    flags.DEFINE_float("learning_rate_g", 2e-4, "Learning rate for the Generator optimizer [2e-4]")
    flags.DEFINE_float("learning_rate_d", 2e-4, "Learning rate for the Discriminator optimizer [2e-4]")
    flags.DEFINE_boolean("decay_learning_rate", False, "Decay learning rate if True [False]")
    flags.DEFINE_float("beta1", 0.5, "1st Momentum term of Adam optimizer [0.5]")
    flags.DEFINE_float("beta2", 0.999, "2nd Momentum term of Adam optimizer [0.999]")

    # flags for running
    flags.DEFINE_string("experiment_name", "unittest_generator", "Name of experiment for current run, if None: random name [None]")
    flags.DEFINE_boolean("train", False, "Train if True, otherwise test [False]")

    # directory Parameters
    flags.DEFINE_string("data_dir", "dataset/train", "Path to datasets directory [data]")
    flags.DEFINE_string("validation_dir", "dataset/val", "Path to datasets directory [data]")
    flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
    flags.DEFINE_string("checkpoint_dir", "checkpoints", "Directory name to save the checkpoints [checkpoint]")
    flags.DEFINE_string("log_dir", "logs_dev_tf", "Path to log for TensorBoard [logs]")

    if tf.VERSION >= '1.5':
        # parse flags
        flags.FLAGS(sys.argv, known_only=True)
        flags.ArgumentParser()

    return flags.FLAGS
