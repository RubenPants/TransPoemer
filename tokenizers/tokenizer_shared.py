"""
tokenizer_shared.py

This file holds the shared Tokenizer class used by all the other tokenizer classes.
"""
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
import torchtext
from torchtext.data import Field
from transformers import AlbertTokenizer

from utils import make_folder


class Tokenizer:
    """ Responsible for the data. """
    
    def __init__(self,
                 albert=None,
                 en=None,
                 device=None,
                 batch_size=64
                 ):
        self.albert = albert if albert else AlbertTokenizer.from_pretrained('albert-base-v1')
        
        if en is None:
            self.EN = Field(tokenize=self.tokenize_unsafe,
                            pad_token=self.albert.pad_token_id,  # AlbertTokenizer's <pad> token
                            unk_token=self.albert.unk_token_id,  # AlbertTokenizer's <unk> token
                            init_token=self.albert.cls_token_id,  # AlbertTokenizer's [CLS] token
                            eos_token=self.albert.sep_token_id,  # AlbertTokenizer's [SEP] token
                            batch_first=True)  # Batch-dimension is first
            
            # Setup EN
            train = torchtext.datasets.TranslationDataset('data/processed/train/',
                                                          exts=("keywords", "complete"),
                                                          fields=(self.EN, self.EN))
            
            # Only use the 20'000 most used tokens (i.e. 2/3rd most used tokens of the albert-vocabulary)
            self.EN.build_vocab(train, max_size=20000)
            self.EN.tokenize = self.tokenize
        else:
            self.EN = en
        
        # Unleash the power of CUDA
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
    
    def __str__(self):
        return f"TokenizerShared(voc_size={len(self.EN.vocab)})"
    
    def tokenize_unsafe(self, text):
        """ Tokenize English text from a string into a list of strings. """
        return self.albert.encode(text, add_special_tokens=False)
    
    def tokenize(self, text):
        """ Tokenize English text from a string into a list of strings. """
        enc = self.albert.encode(text, add_special_tokens=False)
        return [e for e in enc if e in self.EN.vocab.stoi]  # Only keep known words, prevent <unk> from being predicted
    
    def stitch(self, text):
        """ Stitch a separated sentence back together. """
        return self.albert.decode([t for t in text])  # Replace <unk> with 1


def poem_transform_keywords(keywords, model, max_len=70):
    """ Create a poem of the keywords by inferring the model """
    model.eval()
    
    # Transform the keywords to model-readable format
    keywords = [model.tokenizer.EN.init_token] + keywords + [model.tokenizer.EN.eos_token]
    src_indexes = [model.tokenizer.EN.vocab.stoi[token] if token in model.tokenizer.EN.vocab.stoi else
                   model.tokenizer.EN.vocab.stoi[model.tokenizer.EN.unk_token] for token in keywords]
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(model.tokenizer.device)
    src_mask = model.make_src_mask(src_tensor)
    
    # Encode the input via the model's encoder
    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)
    
    # Placeholder for the target (model's prediction)
    trg_indexes = [model.tokenizer.EN.vocab.stoi[model.tokenizer.EN.init_token]]
    
    # Predict token by token until the end-of-sequence token is read (SEP)
    for i in range(max_len):
        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(model.tokenizer.device)
        trg_mask = model.make_trg_mask(trg_tensor)
        
        # Result of current state
        with torch.no_grad():
            output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
        
        # Softmax to find most likely token
        pred_token = output.argmax(2)[:, -1].item()
        trg_indexes.append(pred_token)
        
        # Stop the translation when end-of-sequence token is read
        if pred_token == model.tokenizer.EN.vocab.stoi[model.tokenizer.EN.eos_token]:
            break
    
    # Transform target and return together with final attention layer
    trg_tokens = [model.tokenizer.EN.vocab.itos[i] for i in trg_indexes]
    return trg_tokens[1:], attention


def display_attention(sentence,
                      translation,
                      attention,
                      save_path: str = '',
                      n_heads: int = 8,
                      n_rows: int = 4,
                      n_cols: int = 2,
                      show: bool = True):
    """ Display the attention for each of the heads of the decoder """
    assert n_rows * n_cols == n_heads
    
    # Create new folder to store images (if not yet exist)
    if save_path:
        make_folder(save_path)
    
    # Plot each of the heads
    for i in range(n_heads):
        fig = plt.figure(figsize=(5, 10))
        ax = fig.add_subplot(1, 1, 1)
        _attention = attention.squeeze(0)[i].cpu().detach().numpy()
        ax.matshow(_attention, cmap='bone')
        
        ax.tick_params(labelsize=12)
        ax.set_xticklabels([''] + ['[CLS]'] + [t.lower() for t in sentence] + ['[SEP]'], rotation=90)
        ax.set_yticklabels([''] + translation)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
        fig.tight_layout()
        
        if save_path:
            plt.savefig(save_path + f"/head_{i}")
        if i == 0 and show:  # Only show the first head
            plt.show()
        plt.close()


if __name__ == '__main__':
    os.chdir("..")
    tok = Tokenizer()
    print(tok.tokenize('corona'))
