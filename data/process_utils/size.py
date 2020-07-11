"""
size.py

Processing files related to the size of the samples.
"""
from transformers import AlbertTokenizer

# Parameters
TOKENIZER_TAG = 'albert-base-v1'
TOKENIZER = AlbertTokenizer.from_pretrained(TOKENIZER_TAG)
MAX_LENGTH = 68  # Maximum length of a tokenizer's input sequence
POINT_ENC = 9  # The integer representing the point '.' in the tokenizer
QUESTION_ENC = 60  # The integer representing the point '?' in the tokenizer
EXCL_ENC = 187  # The integer representing the point '!' in the tokenizer
SPLIT_ENC = [POINT_ENC, QUESTION_ENC, EXCL_ENC]  # List of splittable characters


def prune_long_segments(poems):
    """ Prune segments that exceed the maximum length. """
    remove_indices = []
    tok = TOKENIZER
    for index, segment in enumerate(poems):
        if len(tok.encode(segment, add_special_tokens=False)) > MAX_LENGTH:
            remove_indices.append(index)
    
    # Remove bad samples
    for i in remove_indices[::-1]:
        del poems[i]


def split_long_segments(poems, max_length=MAX_LENGTH):
    """ Split segments that exceed a given threshold. Create new data-list since size data may extend. """
    tok = TOKENIZER
    split_data = []
    for d in poems:
        encoded = tok.encode(d,
                             add_special_tokens=False,  # Encode without the use of [SEP]...
                             max_length=float('inf'))  # Encode the full sequence, independent of length
        if len(encoded) > max_length:  # Split if exceeds maximum length
            try:
                split = True
                while split:
                    i = min(max_length, len(encoded))
                    if i < max_length:
                        split = False  # Take the complete remainder and stop the while
                    else:
                        while encoded[i] not in SPLIT_ENC:  # Round to nearest point
                            i -= 1
                            if i < 0: raise IndexError
                    split_data.append(tok.decode(encoded[:i + 1]))
                    encoded = encoded[i + 1:]  # Continue with the remainder
            except IndexError:
                pass  # Single sentence occupies full memory block, discard sample
        elif len(encoded) >= 10:  # Not exceeding maximum length and at least 10 tokens long: add sample
            split_data.append(d)
    return split_data
