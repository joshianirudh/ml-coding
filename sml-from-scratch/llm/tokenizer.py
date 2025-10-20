""""
This file contains an implementation for BPE Tokenizer
@author: Anirudh Joshi
"""

import re
from collections import Counter

def compute_bigram_statistics(token_ids, counter):
    bigram_counter = Counter() if counter is None else counter
    for left, right in zip(token_ids, token_ids[1:]):
        bigram_counter[(left, right)] += 1
    return bigram_counter

def replace_bigram(token_ids, bigram, bigram_id):
    idx = 0
    new_token_ids = []
    while idx < len(token_ids):
        if token_ids[idx : idx + 2] == list(bigram):
            new_token_ids.append(bigram_id)
            idx += 2
        else:
            new_token_ids.append(token_ids[idx])
            idx += 1
    return new_token_ids

class Tokenizer:
    def __init__(self):

        self.vocab = [int.to_bytes(i) for i in range(256)]
        self.merge_rules = {}
        self.pattern = re.compile(r'\s*S+')

    def merge(self, spans):
        """
        One BPE merge step:x
        1) Count all bigrams across all spans.
        2) Pick the most frequent bigram (tie-breaker: natural tuple order).
        3) Create a new vocab token = concat(left bytes + right bytes).
        4) Replace that bigram everywhere in all spans.
        """
        bigram_counter = Counter()
        for t_ids in spans:
            bigram_counter = compute_bigram_statistics(t_ids, bigram_counter)

        t_left, t_right = max(bigram_counter.items(), key=lambda x: x[1])

        new_t = self.vocab[t_left] + self.vocab[t_right]
        new_t_id = len(self.vocab)
        self.vocab.append(new_t)
        bigram = (t_left, t_right)
        self.merge_rules[bigram] = new_t_id

        return [replace_bigram(t_ids, bigram, new_t_id) for t_ids in spans]

    def encode(self, text):







