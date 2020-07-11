"""
english.py

Processing files related to the syntax of the samples.
"""
import re

from langdetect import detect, lang_detect_exception
from transformers import AlbertTokenizer

# Parameters
TOKENIZER_TAG = 'albert-base-v1'
TOKENIZER = AlbertTokenizer.from_pretrained(TOKENIZER_TAG)


def check_english(poems):
    """
    Only english poems are allowed. This check is quite a bottleneck...
    """
    remove_indices = []  # Indices of data-segments that must be removed
    for d_index, d in enumerate(poems):
        try:
            if detect(d) != 'en': remove_indices.append(d_index)
        except lang_detect_exception.LangDetectException:
            remove_indices.append(d_index)  # Remove the ones that can't be parsed
    
    # Remove bad samples
    for i in remove_indices[::-1]: del poems[i]


def check_space(poems, length=20):
    """
    Check if the segment contains at least one space every 'length' (defaults to 20) characters.
    """
    remove_indices = []  # Indices of data-segments that must be removed
    for d_index, d in enumerate(poems):
        for i in range(len(d) // length - 1):
            segment = d[length * i:length * (i + 1)]
            if " " not in segment: remove_indices.append(d_index)
    
    # Remove bad samples
    for i in remove_indices[::-1]: del poems[i]


def handle_raw(poems):
    """Do manual substitutions on the raw data via regex."""
    for d_index, d in enumerate(poems): poems[d_index] = re.sub("\n", " ", d)
    for d_index, d in enumerate(poems): poems[d_index] = re.sub("\t", " ", d)
    return poems


def remove_special_symbols(poems):
    """Remove special symbols based on 'latin' unicode."""
    for d_index, d in enumerate(poems):
        segment = ""
        for c in d:
            try:
                c.encode('latin')
                segment += c
            except UnicodeEncodeError:
                segment += ' '  # Replace with a space
        poems[d_index] = segment


def remove_duplicate_segments(poems):
    """Remove segments as '...' from the poems."""
    for d_index, d in enumerate(poems):
        poems[d_index] = re.sub('[..][.]+', ' ', d)  # Remove .. or ..., but keeps .
        poems[d_index] = re.sub('[??][?]+', '?', d)  # Substitute ?? or ??? by ?, keep single ? as is
        poems[d_index] = re.sub('[!!][!]+', '!', d)  # Substitute !! or !!! by !, keep single ! as is
        
        # Remove text between [...]
        poems[d_index] = re.sub("[\[].*?[\]]", "", d)
