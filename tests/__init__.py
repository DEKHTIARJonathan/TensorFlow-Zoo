#! /usr/bin/python
# -*- coding: utf-8 -*-

from .test_vgg16 import VGG16_Test
from .test_inceptionV4 import InceptionV4_Test
from .test_squeezenet import SqueezeNet_Test
from .test_mobilenet import MobileNet_Test

__all__ = [
    'VGG16_Test',
    'InceptionV4_Test',
    'SqueezeNet_Test',
    'MobileNet_Test',
]
