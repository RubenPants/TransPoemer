"""
tokenizer_main.py

This file holds the Tokenizer class used during final training on the "limerick" dataset, and is used to transform
raw (str) inputs to tokens (int) and back. It also functions as the container for all the data.
"""
import os

import torchtext
from torchtext.data import BucketIterator

from tokenizers.tokenizer_shared import Tokenizer
from utils import drop, prep, read_as_pickle, save_as_pickle


class TokenizerMain(Tokenizer):
    def __init__(self, shared=None, overwrite=False):
        super().__init__(
                albert=shared.albert if shared else None,
                en=shared.EN if shared else None,
                device=shared.device if shared else None,
                batch_size=shared.batch_size if shared else 64,
        )
        
        # Data used for training (overwrites super.train by using safe-tokenizer)
        if not os.path.isfile('tokenizers/storage/valid.pickle') or overwrite:
            prep("Safe-tokenize valid dataset...")
            self.valid = torchtext.datasets.TranslationDataset('data/processed/valid/',
                                                               exts=("keywords", "complete"),
                                                               fields=(self.EN, self.EN))
            save_as_pickle(self.valid.examples, 'tokenizers/storage/valid')
            drop()
        else:
            prep("Loading pickled valid dataset...")
            self.valid = torchtext.datasets.TranslationDataset('data/dummy/',
                                                               exts=('dummy', 'dummy'),
                                                               fields=(self.EN, self.EN))
            self.valid.examples = read_as_pickle('tokenizers/storage/valid')
            drop()
        
        if not os.path.isfile('tokenizers/storage/test.pickle') or overwrite:
            prep("Safe-tokenize test dataset...")
            self.test = torchtext.datasets.TranslationDataset('data/processed/test/',
                                                              exts=("keywords", "complete"),
                                                              fields=(self.EN, self.EN))
            save_as_pickle(self.test.examples, 'tokenizers/storage/test')
            drop()
        else:
            prep("Loading pickled test dataset...")
            self.test = torchtext.datasets.TranslationDataset('data/dummy/',
                                                              exts=('dummy', 'dummy'),
                                                              fields=(self.EN, self.EN))
            self.test.examples = read_as_pickle('tokenizers/storage/test')
            drop()
        
        if not os.path.isfile('tokenizers/storage/train.pickle') or overwrite:
            prep("Safe-tokenize train dataset...")
            self.train = torchtext.datasets.TranslationDataset('data/processed/train/',
                                                               exts=("keywords", "complete"),
                                                               fields=(self.EN, self.EN))
            save_as_pickle(self.train.examples, 'tokenizers/storage/train')
            drop()
        else:
            prep("Loading pickled train dataset...")
            self.train = torchtext.datasets.TranslationDataset('data/dummy/',
                                                               exts=('dummy', 'dummy'),
                                                               fields=(self.EN, self.EN))
            self.train.examples = read_as_pickle('tokenizers/storage/train')
            drop()
        
        # Setup the iterators to provide the data during training, validation, and testing
        self.train_iterator, self.valid_iterator, self.test_iterator = BucketIterator.splits(
                (self.train, self.valid, self.test),
                batch_size=self.batch_size,
                device=self.device)
    
    def __str__(self):
        return f"TokenizerMain(" \
               f"\n\tvocab_size={len(self.EN.vocab)}, " \
               f"\n\tlen_train={len(self.train)}, " \
               f"\n\tlen_valid={len(self.valid)}," \
               f"\n\tlen_test={len(self.test)}," \
               f"\n)"


if __name__ == '__main__':
    os.chdir("..")
    tok = TokenizerMain()
    print(f"Size of training-set: {len(tok.train)}")
    print(f"Size of validation-set: {len(tok.valid)}")
    print(f"Size of test-set: {len(tok.test)}")
    s = tok.train[0].__dict__['trg']
    x = tok.tokenize(s)
    print(x)
    print(tok.albert.decode(x))
