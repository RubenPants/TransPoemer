"""
tokenizer_pre.py

This file holds the Tokenizer class used during pre-training to transform raw (str) inputs to tokens (int) and back. It
also functions as the container for all the data.
"""
import os

import torchtext
from torchtext.data import BucketIterator

from tokenizers.tokenizer_shared import Tokenizer
from utils import drop, prep, read_as_pickle, save_as_pickle


class TokenizerPre(Tokenizer):
    def __init__(self, shared=None, overwrite=False):
        super().__init__(
                albert=shared.albert if shared else None,
                en=shared.EN if shared else None,
                device=shared.device if shared else None,
                batch_size=shared.batch_size if shared else 64,
        )
        
        # Data used for pre-training (overwrites super.train)
        if not os.path.isfile('tokenizers/storage/pre_valid.pickle') or overwrite:
            prep("Safe-tokenize pre_valid dataset...")
            self.valid = torchtext.datasets.TranslationDataset('data/processed/valid/',
                                                               exts=("pre_keywords", "pre_complete"),
                                                               fields=(self.EN, self.EN))
            save_as_pickle(self.valid.examples, 'tokenizers/storage/pre_valid')
            drop()
        else:
            prep("Loading pickled pre_valid dataset...")
            self.valid = torchtext.datasets.TranslationDataset('data/dummy/',
                                                               exts=('dummy', 'dummy'),
                                                               fields=(self.EN, self.EN))
            self.valid.examples = read_as_pickle('tokenizers/storage/pre_valid')
            drop()
        
        if not os.path.isfile('tokenizers/storage/pre_test.pickle') or overwrite:
            prep("Safe-tokenize pre_test dataset...")
            self.test = torchtext.datasets.TranslationDataset('data/processed/test/',
                                                              exts=("pre_keywords", "pre_complete"),
                                                              fields=(self.EN, self.EN))
            save_as_pickle(self.test.examples, 'tokenizers/storage/pre_test')
            drop()
        else:
            prep("Loading pickled pre_test dataset...")
            self.test = torchtext.datasets.TranslationDataset('data/dummy/',
                                                              exts=('dummy', 'dummy'),
                                                              fields=(self.EN, self.EN))
            self.test.examples = read_as_pickle('tokenizers/storage/pre_test')
            drop()
        
        if not os.path.isfile('tokenizers/storage/pre_train.pickle') or overwrite:
            prep("Safe-tokenize pre_train dataset...")
            self.train = torchtext.datasets.TranslationDataset('data/processed/train/',
                                                               exts=("pre_keywords", "pre_complete"),
                                                               fields=(self.EN, self.EN))
            save_as_pickle(self.train.examples, 'tokenizers/storage/pre_train')
            drop()
        else:
            prep("Loading pickled pre_train dataset...")
            self.train = torchtext.datasets.TranslationDataset('data/dummy/',
                                                               exts=('dummy', 'dummy'),
                                                               fields=(self.EN, self.EN))
            self.train.examples = read_as_pickle('tokenizers/storage/pre_train')
            drop()
        
        # Setup the iterators to provide the data during training, validation, and testing
        self.train_iterator, self.valid_iterator, self.test_iterator = BucketIterator.splits(
                (self.train, self.valid, self.test),
                batch_size=self.batch_size,
                device=self.device)
    
    def __str__(self):
        return f"TokenizerPre(" \
               f"\n\tvocab_size={len(self.EN.vocab)}, " \
               f"\n\tlen_train={len(self.train)}, " \
               f"\n\tlen_valid={len(self.valid)}," \
               f"\n\tlen_test={len(self.test)}," \
               f"\n)"


if __name__ == '__main__':
    os.chdir("..")
    tok = TokenizerPre()
    print(f"Size of training-set: {len(tok.train)}")
    print(f"Size of validation-set: {len(tok.valid)}")
    print(f"Size of test-set: {len(tok.test)}")
    s = tok.train[0].__dict__['trg']
    x = tok.tokenize(s)
    print(x)
    print(tok.albert.decode(x))
