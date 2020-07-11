"""
tokens.py

Processing files related to the tokens.
"""

from transformers import AlbertTokenizer

TOKENIZER_TAG = 'albert-base-v1'
TOKENIZER = AlbertTokenizer.from_pretrained(TOKENIZER_TAG)
UNK_ENC = 1  # The integer representing the unknown token '<unk>'


def remove_unk_samples(poems):
    """
    Remove samples that contain <unk> token(s).
    """
    tok = TOKENIZER
    remove_indices = []  # Indices of data-segments that must be removed
    for d_index, d in enumerate(poems):
        if UNK_ENC in tok.encode(d, add_special_tokens=False):
            remove_indices.append(d_index)
    
    # Remove bad samples
    for i in remove_indices[::-1]:
        del poems[i]
