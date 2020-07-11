"""
seq2seq.py

Sequence-to-Sequence model handled by Transformer.
"""
import os

import torch
import torch.nn as nn
from nltk.translate.gleu_score import corpus_gleu
from tqdm import tqdm

from tokenizers.tokenizer_main import TokenizerMain
from tokenizers.tokenizer_pre import TokenizerPre
from tokenizers.tokenizer_shared import poem_transform_keywords, Tokenizer
from transformer.decoder import Decoder
from transformer.encoder import Encoder


class Seq2Seq(nn.Module):
    def __init__(self,
                 name,  # Name of the model
                 overwrite=False,  # Overwrite the tokenizers' data
                 hid_dim=128,  # Dimension of the hidden layer
                 enc_layers=3,  # Number of layers/head of the encoder
                 dec_layers=3,  # Number of layers/head of the decoder
                 enc_heads=8,  # Number of heads in the encoder
                 dec_heads=8,  # Number of heads in the decoder
                 enc_pf_dim=256,  # Point-wise Feedforward dimension in the encoder
                 dec_pf_dim=256,  # Point-wise Feedforward dimension in the encoder
                 enc_dropout=0.1,  # Dropout ratio of the encoder
                 dec_dropout=0.1,  # Dropout ratio of the decoder
                 ):
        super().__init__()
        self.name = name
        
        # Set the used tokenizer
        shared = Tokenizer()
        self.tokenizer = TokenizerMain(shared=shared, overwrite=overwrite)
        self.tokenizer_pre = TokenizerPre(shared=shared, overwrite=overwrite)
        
        self.encoder = Encoder(len(self.tokenizer.EN.vocab),
                               hid_dim,
                               enc_layers,
                               enc_heads,
                               enc_pf_dim,
                               enc_dropout,
                               self.tokenizer.device)
        self.decoder = Decoder(len(self.tokenizer.EN.vocab),
                               hid_dim,
                               dec_layers,
                               dec_heads,
                               dec_pf_dim,
                               dec_dropout,
                               self.tokenizer.device)
        self.src_pad_idx = self.tokenizer.EN.vocab.stoi[self.tokenizer.EN.pad_token]
        self.trg_pad_idx = self.tokenizer.EN.vocab.stoi[self.tokenizer.EN.pad_token]
        self.device = self.tokenizer.device
    
    def __str__(self):
        return self.name if self.name else f'model{f"_{self.version}" if self.version else ""}'
    
    def make_src_mask(self, src):
        # src_mask = [batch size, 1, 1, src len]
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask
    
    def make_trg_mask(self, trg):
        # trg_pad_mask = [batch size, 1, trg len, 1]
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(3)
        trg_len = trg.shape[1]
        
        # trg_sub_mask = [trg len, trg len]
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=self.tokenizer.device)).bool()
        
        # trg_mask = [batch size, 1, trg len, trg len]
        trg_mask = trg_pad_mask & trg_sub_mask
        
        return trg_mask
    
    def forward(self, src, trg):
        # src_mask = [batch size, 1, 1, src len]
        # trg_mask = [batch size, 1, trg len, trg len]
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        
        # enc_src = [batch size, src len, hid dim]
        enc_src = self.encoder(src, src_mask)
        
        # output = [batch size, trg len, output dim]
        # attention = [batch size, n heads, trg len, src len]
        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)
        
        return output, attention


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1: nn.init.xavier_uniform_(m.weight.data)


def train(model, iterator, optimizer, criterion, clip=1):
    model.train()  # Put train mode to True
    epoch_loss = 0
    
    for i, batch in tqdm(enumerate(iterator), total=len(iterator)):
        src = batch.src
        trg = batch.trg
        
        optimizer.zero_grad()
        
        # output = [batch size, trg len - 1, output dim]
        output, _ = model(src, trg[:, :-1])
        
        output_dim = output.shape[-1]
        
        # output = [batch size * trg len - 1, output dim]
        output = output.contiguous().view(-1, output_dim)
        
        # trg = [batch size * trg len - 1]
        trg = trg[:, 1:].contiguous().view(-1)
        
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
    
    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    
    with torch.no_grad():
        for i, batch in tqdm(enumerate(iterator), total=len(iterator)):
            src = batch.src
            trg = batch.trg
            
            # output = [batch size, trg len - 1, output dim]
            output, _ = model(src, trg[:, :-1])
            output_dim = output.shape[-1]
            
            # output = [batch size * trg len - 1, output dim]
            output = output.contiguous().view(-1, output_dim)
            
            # trg = [batch size * trg len - 1]
            trg = trg[:, 1:].contiguous().view(-1)
            
            loss = criterion(output, trg)
            epoch_loss += loss.item()
    
    return epoch_loss / len(iterator)


def get_cuda_device():
    """ Get CUDA enable device. """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def calculate_gleu(model, data, max_len=70):
    """ Calculate the GLEU score of the tokenizer's full test-set. """
    trgs = []
    pred_trgs = []
    for sample in tqdm(data, desc="Making predictions"):
        src = sample.src
        trg = sample.trg
        
        pred_trg, _ = poem_transform_keywords(
                keywords=src,
                model=model,
                max_len=max_len
        )
        
        # cut off <eos> token
        pred_trg = pred_trg[:-1]
        
        pred_trgs.append(pred_trg)
        trgs.append([trg])
    return gleu_score(trgs, pred_trgs)


def gleu_score(refs, pred):
    """
    Calculate the corpus GLEU score.
    For more information: https://www.nltk.org/api/nltk.translate.html

    :param refs: References - list(list(list(str|int)))
    :param pred: Predictions - list(list(str|int))
    :return: GLEU-score
    """
    score = corpus_gleu(refs, pred)
    return round(score * 100, 2)


if __name__ == '__main__':
    # Go back to root directory
    os.chdir("..")
    
    # Load model
    model = Seq2Seq('model').to(get_cuda_device())
    params = model.parameters()
