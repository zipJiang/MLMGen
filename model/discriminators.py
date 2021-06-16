"""This file contains definition of a list of
discriminators that could be used with fudge.
"""
import torch
from transformers import AutoModel, AutoTokenizer, BartTokenizer, BartTokenizerFast, BartForConditionalGeneration
from typing import Text, List, Dict, Union, Optional, Any
from abc import ABC, abstractmethod


class DiscriminatorFudge(ABC):
    def __init__(self, batch_size: int = 64):
        self.batch_size = batch_size
        
    @abstractmethod
    def __call__(self, context: List[Text], word: Text) -> torch.Tensor:
        """
        """
        pass
    
    def form_batches(self, *args):
        """Here we use the batch_size hyper to split the
        original data into batches.
        """
        length = -1
        for item in args:
            assert isinstance(item, torch.Tensor), "Input is not of torch.Tensor type!"
            assert length == -1 or item.shape[0] == length, "Item of unequal size!"
            length = item.shape[0]
            
        for i in range(0, length, self.batch_size):
            yield (item[i:i + self.batch_size] for item in args)
    
    
class BartDiscriminatorFudge(DiscriminatorFudge):
    def __init__(self, context_name: Text, tokenizer_name: Text,
                 batch_size: int = 64, device: Text = 'cpu',
                 agg_func: Text = 'mean'):
        super().__init__(batch_size)
        self.device = device
        self.model = BartForConditionalGeneration.from_pretrained(context_name)
        self.model.eval()
        self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.agg_func = agg_func
        
    def __call__(self, context: List[Text], glosses: List[Text], truth_label: int,
                 return_dist: bool = False) -> torch.Tensor:
        """Now we are doing the WSD. Return the selection dataset of the data.
        """
        tokenized = self.tokenizer(context,
                                   add_special_tokens=True,
                                   # max_length is now fixed to 128
                                   max_length=64,
                                   truncation=True,
                                   return_attention_mask=True,
                                   return_tensors='pt',
                                   padding='max_length')

        tokenized = dict(tokenized)
        
        # also we need to tokenize the gloss as the output.
        gloss_tokenized = self.tokenizer(
            glosses,
            add_special_tokens=True,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
            max_length=64
        )
        
        def _to_labels(input_ids: torch.Tensor, pad_token_id: int, repeat: int) -> torch.Tensor:
            """
            """
            return input_ids.masked_fill(input_ids == pad_token_id, -100).repeat(repeat, 1)

        # now we combine them together and form a complete batch.
        
        losses_by_choice = []
        
        for i in range(gloss_tokenized['input_ids'].shape[0]):
            labels = _to_labels(gloss_tokenized['input_ids'][i:i + 1], pad_token_id=self.tokenizer.pad_token_id, repeat=tokenized['input_ids'].shape[0])
            labels = labels.to(self.device)
            decoder_dict = {
                'decoder_attention_mask': gloss_tokenized['attention_mask'][i:i + 1].expand(tokenized['input_ids'].shape[0], -1),
                'decoder_input_ids': self.model.prepare_decoder_input_ids_from_labels(labels).expand(tokenized['input_ids'].shape[0], -1)
            }
            
            # Notice that python update function will rewrite the original value.
            tokenized.update(decoder_dict)
            tokenized = {key: val.to(self.device) for key, val in tokenized.items()}
            
            loss_func = torch.nn.CrossEntropyLoss(reduction='none')
            
            with torch.no_grad():
                outputs = self.model(**tokenized)
                # we are not inserting labels so that we manually compute the loss.
                lm_logits = outputs.logits.view(-1, outputs.logits.shape[-1])
                labels = labels.view(-1)
                
                loss_ = loss_func(lm_logits, labels)
                loss_ = loss_.view(-1, 64)
                
                batch_loss = - getattr(torch, self.agg_func)(loss_, dim=-1)
                losses_by_choice.append(batch_loss)
                
        losses_by_choice = torch.stack(losses_by_choice, dim=-1)
        losses_by_choice = losses_by_choice - torch.logsumexp(losses_by_choice, dim=-1, keepdim=True)
        
        if not return_dist:
            losses_by_choice = losses_by_choice[:, truth_label]
        
        # return the log-likelihood of the losses.
        return losses_by_choice
    
class BartGenDiscriminatorFudge(DiscriminatorFudge):
    def __init__(self, context_name: Text, tokenizer_name: Text,
                 batch_size: int = 64, device: Text = 'cpu',
                 agg_func: Text = 'mean'):
        """This discriminator only evaluate one certain condition instead of running
        all glosses through.
        """
        super().__init__(batch_size)
        self.device = device
        self.model = BartForConditionalGeneration.from_pretrained(context_name)
        self.model.eval()
        self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.agg_func = agg_func
        
    def __call__(self, context: List[Text], gloss: List[Text]) -> torch.Tensor:
        assert len(context) == len(gloss), 'The context and gloss inputs are not of the same length.'
        
        tokenized = self.tokenizer(context,
                                   add_special_tokens=True,
                                   # max_length is now fixed to 128
                                   max_length=64,
                                   truncation=True,
                                   return_attention_mask=True,
                                   return_tensors='pt',
                                   padding='max_length')
        
        gloss_tokenized = self.tokenizer(
            gloss,
            add_special_tokens=True,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
            max_length=64
        )
        
        # combine them together with labels
        def _to_labels(input_ids: torch.Tensor, pad_token_id: int) -> torch.Tensor:
            """
            """
            return input_ids.masked_fill(input_ids == pad_token_id, -100)
        
        labels = _to_labels(gloss_tokenized['input_ids'], pad_token_id=self.tokenizer.pad_token_id)
        tokenized['decoder_input_ids'] = self.model.prepare_decoder_input_ids_from_labels(labels)
        tokenized['decoder_attention_mask'] = gloss_tokenized['attention_mask']
        
        loss_func = torch.nn.CrossEntropyLoss(reduction='none')
        
        tokenized = {k: v.to(self.device) for k, v in tokenized.items()}
        labels = labels.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**tokenized)
            # we are not inserting labels so that we manually compute the loss.
            lm_logits = outputs.logits.view(-1, outputs.logits.shape[-1])
            labels = labels.view(-1)
            
            loss_ = loss_func(lm_logits, labels)
            loss_ = loss_.view(-1, 64)
            
            gloss_likelihood = - getattr(torch, self.agg_func)(loss_, dim=-1)
            
        return gloss_likelihood