import unittest
from messages_completion.samplers import create_sampler
from messages_completion.samplers.factory import get_available_methods


class TestSamplerFactory(unittest.TestCase):
    def test_get_available_methods_only_dcbs(self):
        self.assertEqual(get_available_methods(), ["dcbs"])

    def test_create_dcbs_sampler(self):
        sampler = create_sampler("dcbs", {"k": 4, "top_n": 20})
        # DCBS sampler may require dependencies; just assert that factory returns something non-None
        # or raises ImportError depending on environment
        if sampler is None:
            self.fail("DCBS sampler factory returned None for 'dcbs'")

    def test_create_standard_returns_none(self):
        self.assertIsNone(create_sampler("greedy"))
        self.assertIsNone(create_sampler("top_p", {"p": 0.9}))
        self.assertIsNone(create_sampler("nucleus", {"p": 0.8}))
