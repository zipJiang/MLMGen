"""Adding debugging utilities for the model.
"""
import torch
import transformers


def pprint_categorical_dist(tokenizer: transformers.PreTrainedTokenizer,
                            logits: torch.Tensor, ids: torch.Tensor):
    """This function examine the relationship between logits and
    tokens.
    """
    # normalize logits
    logits = logits[0]
    ids = ids[0]
    tokens = tokenizer.convert_ids_to_tokens(ids.cpu().tolist())
    logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    probs = torch.exp(logits)
    
    # we only debug the first line of data.
    for pb, token in zip(probs, tokens):
        print(pb, token)