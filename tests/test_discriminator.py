"""Run unittest over the generativeWSD
discriminator.
"""
import unittest
from model.discriminators import BartGenDiscriminatorFudge
from typing import Text, Dict, Any
import torch
import json
from sklearn.metrics import accuracy_score
import random
from tqdm import tqdm


class TestDiscriminator(unittest.TestCase):
    def setUp(self):
        """
        """
        self.discriminator = BartGenDiscriminatorFudge(context_name='fudge_ckpt/generative_wsd_ckpt/checkpoint-114795',
                                                       tokenizer_name='fudge_ckpt/generative_wsd_ckpt/checkpoint-114795',
                                                       device='cuda:3')
        
    def test_simple_discrimination(self):
        test_cases = {'gloss': ['The shore of a river, lake, or other body of water; the land along the edge of a lake, valley, or valley.'] * 4,
                      'context': [
                                    'I sit on the river <WSD>bank</WSD>.',
                                    'I save money to the international <WSD>bank</WSD>.',
                                    'He save money to the international <WSD>bank</WSD>.',
                                    'To the <WSD>bank</WSD> we go, to see the beautiful lake.']}
        
        result = self.discriminator(**test_cases).cpu()
        result = torch.exp(result)
        
        for c, r in zip(test_cases['context'], result):
            print(c, r)