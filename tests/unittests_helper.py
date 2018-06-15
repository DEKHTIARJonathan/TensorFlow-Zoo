#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
from unittest.util import strclass

from contextlib import contextmanager

__all__ = [
    'CustomTestCase',
]


class CustomTestCase(unittest.TestCase):
    @classmethod
    def __str__(cls):
        return strclass(cls)

    @classmethod
    def __unicode__(cls):
        return strclass(cls)

    @classmethod
    def __repr__(cls):
        return strclass(cls)

    @contextmanager
    def assertNotRaises(self, exc_type):
        try:
            yield None
        except exc_type:
            raise self.failureException('{} raised'.format(exc_type.__name__))

    @classmethod
    def setUpClass(cls):

        try:
            super(CustomTestCase, cls)._set_up(cls)
        except AttributeError:
            try:
                cls._set_up(cls)
            except AttributeError:
                pass

    @classmethod
    def tearDownClass(cls):
        try:
            super(CustomTestCase, cls)._tear_down(cls)
        except AttributeError:
            try:
                cls._tear_down(cls)
            except AttributeError:
                pass
