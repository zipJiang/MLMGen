"""This module is intended to build a underlining MLM
agnostic Gibbs sampler for the sentence generation task.
"""
import transformers
from transformers import AutoModelForMaskedLM, AutoTokenizer, PreTrainedTokenizerFast, PreTrainedModel
from typing import Callable, Tuple, Dict, Optional, Text, List
from dataclasses import dataclass, field
import torch
import numpy as np
from tqdm import tqdm
from functools import partial
import random
from abc import ABC, abstractmethod


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
    batch_size: int = field(default=32,
                            metadata={
                                'help': "batch size to be used by sampler."})
    device: Text = field(default='cuda:2',
                         metadata={'help': "Which device to use."})
    ckpt_name: Text = field(default='bert-base-uncased',
                            metadata={'help': 'Where to load the unline model'})
    cache_dir: Text = field(default='cache/',
                            metadata={'help': 'Where to store the downloaded model.'})
    
    
class MLMSamplerBase(ABC):
    """We are refactoring the generation script into OOP,
    thus we need to define some base class object.
    """
    def __init__(self, configuration: ConfigurationMLMSampler):
        """We assume that for model the return_dict is set to 'True'
        """
        self.configuration = configuration
        self.model, self.tokenizer = get_model(self.configuration.ckpt_name, self.configuration.device, self.configuration.cache_dir)
        
    def sample_sentences(self, num_samples: int,
                         word: Text = ''):
        """
        """
        lengths = torch.randint(low=self.configuration.length_range[0],
                                high=self.configuration.length_range[1],
                                dtype=torch.int64, size=(num_samples,))
        lengths = lengths.to(self.configuration.device)
        # construct sentences of length in lengths with tokenizer.batch_encode_plus
        token_ids, attention_mask, forbidden_ids = self._create_template(lengths, word)

        decoded_sentences = []

        # first iterate through batches:
        for i in tqdm(range(0, num_samples, self.configuration.batch_size)):
            batch_input, batch_mask = token_ids[i:i + self.configuration.batch_size],\
                attention_mask[i:i + self.configuration.batch_size]
            batch_lengths = lengths[i:i + self.configuration.batch_size]

            fbids = forbidden_ids[i: i + self.configuration.batch_size]

            for ridx in range(self.configuration.sampling_rounds):
                pos = self._sample_pos(batch_lengths, fbids)
                batch_input = self._generation_step(batch_input,
                                                    batch_mask,
                                                    pos,
                                                    burn_in=ridx < self.configuration.burn_in_rounds)

            # after the sampling, do decoding
            decoded = self.tokenizer.batch_decode(batch_input.cpu().tolist(),
                                                  skip_special_tokens=True,
                                                  clean_up_tokenization_spaces=True)

            decoded_sentences.extend(decoded)

        return decoded_sentences
    
    def _generation_step(self, input_ids: torch.Tensor,
                         attention_mask: torch.Tensor,
                         pos: torch.Tensor,
                         burn_in: bool = False):
        """Can access masked tokens as self.tokenizer.mask_token,
        self.tokenizer.cls_token, self.tokenizer.sep_token.

        etc.

        input_ids, attention_mask: --- [batch_size, seq_len]
        pos: --- [batch_size]
        """
        # iterative = torch.arange(pos.shape[0]).long()
        # input_ids[iterative, pos] = self.tokenizer.mask_token_id
        # outputs = self.mlm_model(input_ids, attention_mask)
        # logits = outputs.logits[iterative, pos]
        
        with torch.no_grad():
            
            logits = self._get_logits(input_ids, attention_mask, pos)
            samples = self._sample_from_logits(logits, burn_in=burn_in)
            
        return self._gather_preds(input_ids, samples, pos)

    @abstractmethod
    def _create_template(self, lengths: torch.Tensor,
                         word: Text) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """return:
            [TEMPLATE_TENSOR, ATTENTION_MASK, FORBIDDEN_IDS]
        """
        pass
    
    @abstractmethod
    def _get_logits(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """Returns the logits of the given positions.
        """
        pass
    
    @abstractmethod
    def _gather_preds(self, input_ids: torch.Tensor, samples: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """Gather model prediction and generate final output of this step.
        """
        pass

    def _sample_pos(self, lengths: torch.Tensor, fbids: List[int]) -> torch.Tensor:
        """Sample prediction positions for the sentence tensors.
        """
        pass

    def _sample_from_logits(self, logits: torch.Tensor, burn_in: bool) -> torch.Tensor:
        """Given positional logits and sample final predictions.
        """
        pass


class SimpleMLMSampler(MLMSamplerBase):
    """This function is a Sampler without any additional features.
    """
    def __init__(self, configuration: ConfigurationMLMSampler):
        """In our model we will use top_k, batch_size, etc.
        """
        super().__init__(configuration)


    def _sample_from_logits(self, logits: torch.Tensor,
                            burn_in: Optional[bool])\
            -> torch.Tensor:
        """The difference compared with previous implementation is
        that we now will by default use the top_k setted during initialization.
        """
        if not burn_in:
            lgt, ids = torch.topk(logits / self.configuration.temperature, k=self.configuration.top_k, dim=-1)
        else:
            lgt = logits

        # samples = torch.multinomial(torch.softmax(lgt, dim=-1),
        #                             num_samples=1,
        #                             replacement=True)
        sampler = torch.distributions.categorical.Categorical(logits=lgt)
        samples = sampler.sample()
        samples = samples.unsqueeze(-1)
        
        if not burn_in:
            samples = torch.gather(ids, dim=-1, index=samples)

        # samples [batch_size]
        return samples.squeeze(-1)

    def _create_template(self, lengths: torch.Tensor,
                         word: Text = ''):
        """
        """

        word_tknz = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.tokenize(word))
        word_tknz = torch.tensor(word_tknz, dtype=torch.int64)

        # create a candidate list
        template = torch.zeros(lengths.shape[0], self.configuration.length_range[1],
                               dtype=torch.int64)
        template.fill_(self.tokenizer.mask_token_id)
        forbidden_mask = torch.ones(lengths.shape[0], self.configuration.length_range[1],
                                    dtype=torch.float32)

        pos_word_begin = self._sample_pos(lengths - len(word_tknz))
        
        # TODO: change forbidden_ids into a mask might be better.
        for lidx, (length, pb) in enumerate(zip(lengths, pos_word_begin)):
            # sample sentence one-by-one
            template[lidx, 0] = self.tokenizer.cls_token_id
            template[lidx, length - 1] = self.tokenizer.sep_token_id
            template[lidx, length:] = self.tokenizer.pad_token_id
            template[lidx, pb:pb + len(word_tknz)] = word_tknz
            forbidden_mask[lidx, pb:pb + len(word_tknz)] = 0.
            forbidden_mask[lidx, 0] = 0.
            forbidden_mask[lidx, length - 1] = 0.
            
        print(template)
            
        attention_mask = torch.ones_like(template)
        attention_mask.masked_fill_(template == self.tokenizer.pad_token_id, 0)
            
        return template.to(self.configuration.device), attention_mask.to(self.configuration.device), forbidden_mask.to(self.configuration.device)

    def _sample_pos(self, lengths: torch.Tensor, forbidden: Optional[torch.Tensor] = None):
        """Sample a position to be updated
        """
        categorical = torch.ones((lengths.shape[0], self.configuration.length_range[1]),
                                 dtype=torch.float32)
        categorical = categorical.to(self.configuration.device)

        for i, length in enumerate(lengths):
            categorical[i, length:] = 0.
            if forbidden is not None:
                categorical = categorical * forbidden

        # renormalize categorical
        categorical /= torch.sum(categorical, dim=-1, keepdim=True)
        pos = torch.multinomial(categorical, num_samples=1,
                                replacement=True)

        # pos [batch_size]
        
        return pos.squeeze(-1).to(self.configuration.device)
    
    def _get_logits(self, input_ids: torch.Tensor,
                    attention_mask: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """This function gather the logits at given location.
        """
        iterator = torch.arange(0, input_ids.shape[0])
        
        outputs = self.model(input_ids, attention_mask)
        logits = outputs.logits
        logits = logits[iterator, pos]

        return logits  # of shape [self.batch_size, num_logits]

    def _gather_preds(self, input_ids: torch.Tensor, samples: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """Function for gathering the precdiction for samples.
        
        samples: [self.configuration.batch_size]
        input_ids: [self.configuration.batch_size, seq_len]
        pos: [self.configuration.batch_size]
        """
        
        iterator = torch.arange(0, input_ids.shape[0])
        input_ids[iterator, pos] = samples
        
        return input_ids


MODEL_NAME_DICT = {
    'bert': 'bert-base-uncased',
    'electra': 'google/electra-large-generator'
}

    
def get_model(model_name: Text,
              device: Text,
              cache_dir: Text) -> Tuple[PreTrainedModel, PreTrainedTokenizerFast]:
    """This function will get us a model of
    the model to be used for the function.
    """
    model = AutoModelForMaskedLM.from_pretrained(model_name, cache_dir=cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    
    model.to(device)
    model.eval()
    
    return model, tokenizer