#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
from unittest.util import strclass

from contextlib import contextmanager

from abc import ABC, abstractmethod

__all__ = [
    'Abstract_TestCase',
    'CustomTestCase',
]


class Abstract_TestCase(ABC):

    @abstractmethod
    def _ANO_MODEL(self): raise NotImplementedError()

    def assertEqual(self, expr, msg=None): raise NotImplementedError()

    def assertEquals(self, expr, msg=None): raise NotImplementedError()

    def assertNotEqual(self, expr, msg=None): raise NotImplementedError()

    def assertNotEquals(self, expr, msg=None): raise NotImplementedError()

    def assertAlmostEqual(self, expr, msg=None): raise NotImplementedError()

    def assertAlmostEquals(self, expr, msg=None): raise NotImplementedError()

    def assertNotAlmostEqual(self, expr, msg=None): raise NotImplementedError()

    def assertNotAlmostEquals(self, expr, msg=None): raise NotImplementedError()

    def assertRaises(self, expr, msg=None): raise NotImplementedError()

    def assertNotRaises(self, expr, msg=None): raise NotImplementedError()

    def assertTrue(self, expr, msg=None): raise NotImplementedError()

    def assertFalse(self, expr, msg=None): raise NotImplementedError()


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
