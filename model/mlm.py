"""This module is intended to build a underlining MLM
agnostic Gibbs sampler for the sentence generation task.
"""
import transformers
from typing import Callable, Tuple, Dict, Optional, Text, List
from dataclasses import dataclass, field
import torch
import numpy as np
from tqdm import tqdm
from functools import partial
import random


@dataclass
class ConfigurationMLMSampler:
    """Configuraion for the MLMSampler.
    """
    length_range: Tuple[int, int] = field(default=(30, 40),
                                          metadata={
                                              'help': "range of generation excluding special tokens."})
    sampling_rounds: int = field(default=500,
                                metadata={
                                    'help': "number of gibbs round for generating sentences. (After mixed)."})
    burn_in_rounds: int = field(default=100,
                                metadata={
                                    'help': "number of burn-in step."})
    temperature: float = field(default=.9,
                               metadata={
                                   'help': "temperature for annealing the sampling distribution."})
    top_k: int = field(default=100,
                       metadata={
                           'help': "sample from only top_k item."})
    top_p: float = field(default=.92,
                         metadata={
                             'help': "Whether we use top_p for nucleus sampling."
                         })
    is_nucleus: bool = field(default=True,
                             metadata={
                                 'help': "Is nucleus sampling applied in the sampling?"
                             })
    batch_size: int = field(default=32,
                            metadata={
                                'help': "batch size to be used by sampler."})
    device: Text = field(default='cuda:2',
                         metadata={'help': "Which device to use."})


class MLMSampler:
    """
    """
    def __init__(self, configuration: ConfigurationMLMSampler,
                 model_config: Callable[[Text], Tuple]):
        """We assume that for model the return_dict is set to 'True'
        """
        self.device = configuration.device
        self.mlm_model, self.tokenizer =\
            model_config(device=self.device)
        self.length_range = configuration.length_range
        self.sampling_rounds = configuration.sampling_rounds
        self.burn_in_rounds = configuration.sampling_rounds
        self.tau = configuration.temperature
        self.top_k = configuration.top_k
        self.top_p = configuration.top_p
        self.batch_size = configuration.batch_size
        self.is_nucleus = configuration.is_nucleus

    def _sample_from_logits(self, logits: torch.Tensor,
                            top_k: Optional[int] = None)\
            -> torch.LongTensor:
        """Sample from top_k elements in a categorical distribution characterized by logits.
        Notice that we are not using self.top_k but a passed-in argument top_k.
        """
        if top_k is not None:
            lgt, ids = torch.topk(logits / self.tau, k=top_k, dim=-1)
        else:
            lgt, _ = logits, None

        samples = torch.multinomial(torch.softmax(lgt, dim=-1),
                                    num_samples=1,
                                    replacement=True)
        if top_k is not None:
            samples = torch.gather(ids, dim=-1, index=samples)

        # samples [batch_size]
        return samples.squeeze(-1)

    def _nucleus_sample_from_logits(self, logits: torch.Tensor,
                                    top_p: Optional[float] = None) -> torch.LongTensor:
        """Sample from top_p probability nucleus instead of
        sample from full posterior dist.
        """
        if top_p is not None:
            assert 0 < top_p and top_p < 1
            logits, indices = torch.sort(logits, descending=True, dim=-1)
            cum_probs = torch.cumsum(logits, dim=-1)

            masked_probs = torch.where(cum_probs > top_p, logits / self.tau, torch.ones_like(logits) * -1e8)
        else:
            masked_probs = logits

        samples = torch.multinomial(torch.softmax(masked_probs, dim=-1),
                                    num_samples=1, replacement=True)

        if top_p is not None:
            samples = torch.gather(indices, dim=-1, index=samples)

        return samples.squeeze(-1)

    def _generation_step(self, input_ids: torch.Tensor,
                         pos: torch.Tensor,
                         attention_mask: Optional[torch.Tensor] = None,
                         burn_in: bool = False):
        """Can access masked tokens as self.tokenizer.mask_token,
        self.tokenizer.cls_token, self.tokenizer.sep_token.

        etc.

        input_ids, attention_mask: --- [batch_size, seq_len]
        pos: --- [batch_size]
        """
        seq_len = input_ids.shape[-1]
        iterative = torch.arange(pos.shape[0]).long()
        input_ids[iterative, pos] = self.tokenizer.mask_token_id

        outputs = self.mlm_model(input_ids, attention_mask)
        logits = outputs.logits[iterative, pos]

        if self.is_nucleus:
            samples = self._nucleus_sample_from_logits(logits,
                                                       top_p=self.top_p if not burn_in else None)
        else:
            samples = self._sample_from_logits(logits,
                                               top_k=self.top_k if not burn_in else None)
        input_ids[iterative, pos] = samples

        return input_ids

    def _create_template(self, lengths: torch.Tensor,
                         word: Text = '', triggers: List[Text] = []):
        """This function create template with padding and attention_mask.
        We use triggers to ground word meaning.
        """

        # we first need to tokenize triggers.
        triggers = [self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(wd))
                    for wd in triggers]
        word_tknz = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.tokenize(word))

        texts = []
        forbidden_ids = []

        # create a candidate list
        candidates = set(range(self.tokenizer.vocab_size))
        candidates -= set([self.tokenizer.mask_token_id,
                           self.tokenizer.cls_token_id,
                           self.tokenizer.sep_token_id,
                           self.tokenizer.pad_token_id])
        candidates = list(candidates)

        for length in lengths:
            length = length.cpu().item()
            random.shuffle(triggers)
            bag = [word_tknz]
            bag_size = len(word_tknz)
            pointer = 0

            while bag_size <= length - 3 and pointer != len(triggers):
                bag.append(triggers[pointer])
                bag_size += len(triggers[pointer])
                pointer += 1

            # now pack bag with [MASK]
            # we are not packing with mask tokens but random tokens.
            bag.extend([random.sample(candidates, k=1) for i in range(length - 3 - bag_size)])

            id_iter = list(range(len(bag)))
            random.shuffle(id_iter)

            text = [self.tokenizer.cls_token_id]
            for seq_id in id_iter:
                if seq_id == 0:
                    #  print(self.tokenizer.convert_ids_to_tokens(bag[seq_id]))
                    fbids = [len(text) + i for i in range(len(word_tknz))] + [0, length - 1, length - 2]
                    forbidden_ids.append(fbids)
                text.extend(bag[seq_id])

            texts.append((text + self.tokenizer.convert_tokens_to_ids(['.'])
                          + [self.tokenizer.sep_token_id]
                          + [self.tokenizer.pad_token_id] * self.length_range[1])[:self.length_range[1]])

        # construct input_ids and attention_mask
        input_ids = torch.tensor(texts, dtype=torch.int64)
        attention_mask = torch.where(input_ids != self.tokenizer.pad_token_id,
                                     torch.ones_like(input_ids), torch.zeros_like(input_ids))

        return {'input_ids': input_ids,
                'attention_mask': attention_mask}, forbidden_ids

    def _sample_pos(self, lengths: torch.Tensor, forbidden: Optional[List[List[int]]] = None):
        """Sample a position to be updated
        """
        categorical = torch.ones((lengths.shape[0], torch.max(lengths)), dtype=torch.float32)
        for i, length in enumerate(lengths):
            categorical[i, length:] = 0.
            if forbidden is not None:
                categorical[i, forbidden[i]] = 0.

        # renormalize categorical
        categorical /= torch.sum(categorical, dim=-1, keepdim=True)
        pos = torch.multinomial(categorical, num_samples=1,
                                replacement=True)

        # pos [batch_size]
        return pos.squeeze(-1)

    def sample_sentences(self, num_samples: int,
                         word: Text = '', triggers: List[Text] = []):
        """
        """
        lengths = torch.randint(low=self.length_range[0], high=self.length_range[1], dtype=torch.int64, size=(num_samples,))
        # construct sentences of length in lengths with tokenizer.batch_encode_plus
        tokenized, forbidden_ids = self._create_template(lengths, word, triggers)

        decoded_sentences = []

        # first iterate through batches:
        for i in tqdm(range(0, num_samples, self.batch_size)):
            batch_input, batch_mask = tokenized['input_ids'][i:i + self.batch_size].to(self.device),\
                tokenized['attention_mask'][i:i + self.batch_size].to(self.device)
            batch_lengths = lengths[i:i + self.batch_size]

            fbids = forbidden_ids[i: i + self.batch_size]

            for ridx in range(self.sampling_rounds):
                pos = self._sample_pos(batch_lengths, fbids)
                batch_input = self._generation_step(batch_input, pos,
                                                    batch_mask,
                                                    burn_in=ridx < self.burn_in_rounds)

            # after the sampling, do decoding
            decoded = self.tokenizer.batch_decode(batch_input.cpu().tolist(),
                                                  skip_special_tokens=True,
                                                  clean_up_tokenization_spaces=True)

            decoded_sentences.extend(decoded)

        return decoded_sentences, forbidden_ids


def get_roberta(device: Text):
    """This is the classical function for getting a automodel.

    Notice that pre_tokenized and there is a bug for adding space. We have to concatenate it and do sampling again for BPE.
    """
    model = transformers.RobertaForMaskedLM.from_pretrained('roberta-base',
                                                            return_dict=True)
    model_config = transformers.RobertaConfig.from_pretrained('roberta-base')
    model.lm_head.decoder.bias = torch.nn.Parameter(torch.zeros(model_config.vocab_size))
    #  config = transformers.RObertaConfig.from_pretrained('roberta-base')
    tokenizer = transformers.RobertaTokenizerFast.from_pretrained(
        #  'roberta-base', add_prefix_space=True)
        'roberta-base')

    model.to(device)
    model.eval()

    return model, tokenizer


def get_bert(device: Text):
    model = transformers.BertForMaskedLM.from_pretrained('bert-base-uncased',
                                                         return_dict=True)
    model_config = transformers.BertConfig.from_pretrained('bert-base-uncased')
    #  model.cls.predictions.decoder.bias = torch.nn.Parameter(torch.zeros(model_config.vocab_size))
    #  config = transformers.RObertaConfig.from_pretrained('roberta-base')
    tokenizer = transformers.BertTokenizerFast.from_pretrained(
        'bert-base-uncased')

    model.to(device)
    model.eval()

    return model, tokenizer


def get_electra(device: Text):
    model = transformers.ElectraForMaskedLM.from_pretrained('google/electra-large-generator', return_dict=True)
    model_config = transformers.ElectraConfig.from_pretrained('google/electra-large-generator')
    #  model.cls.predictions.decoder.bias = torch.nn.Parameter(torch.zeros(model_config.vocab_size))
    #  config = transformers.RObertaConfig.from_pretrained('roberta-base')
    tokenizer = transformers.ElectraTokenizerFast.from_pretrained(
        'google/electra-large-generator')

    model.to(device)
    model.eval()

    return model, tokenizer


if __name__ == '__main__':
    """Simple test for functionality.
    """
    config = ConfigurationMLMSampler()

    sampler = MLMSampler(config, get_bert)
    results, _ = sampler.sample_sentences(10, 'bank',
                                          #  triggers=['river', 'slope', 'water',
                                          #            'tree', 'grass'])
                                          triggers=[])

    for sentence in results:
        print('-' * 20)
        print(sentence)
        print('-' * 20)
