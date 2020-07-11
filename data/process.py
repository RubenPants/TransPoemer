"""
process.py

Read the raw datafiles from the '/raw' folder, transform them to a shared suitable format and place them in the
'/process' sub-folder.

Decisions made
    - Foundation2 is not taken into account since it is only a very small dataset which varies a lot between samples.
      Hence it is not worth the time processing this.
   
Notes:
    - The tokenizer (AlbertTokenizer) has the tendency to display quite some warnings if the threshold is exceeded,
      which doesn't matter (and even is advisable) during pre-processing, so just ignore this.
"""
import argparse
import csv
import os

from data.process_utils.english import check_english, check_space, handle_raw, remove_special_symbols, \
    remove_duplicate_segments
from data.process_utils.keyword import extract_keywords, prune_empty_keywords
from data.process_utils.size import prune_long_segments, split_long_segments
from data.process_utils.tokens import remove_unk_samples
from data.visualize import main as visualize
from utils import drop, make_folder, prep, read_as_file, save_as_file


def read_limericks():
    """ :return: List of poems """
    with open(f'data/raw/limericks.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='"')
        return [r[0] for r in csv_reader]


def process(poems):
    """
    Process the poems.
    
    :param poems: List of raw poems
    :return: List of processed poems
    """
    handle_raw(poems)
    print(f"--> Raw tokens removed")
    remove_special_symbols(poems)
    print(f"--> Special symbols removed")
    remove_duplicate_segments(poems)
    print(f"--> Duplicate segments removed")
    split = split_long_segments(poems)
    split_size = len(split)
    print(f"--> Created {split_size - len(poems)} more samples after split")
    check_space(split)
    space_size = len(split)
    print(f"--> Removed {split_size - space_size} samples that violated the space-constraint")
    remove_unk_samples(split)
    unk_size = len(split)
    print(f"--> Removed {space_size - unk_size} samples that contained '<unk>' symbols")
    check_english(split)
    english_size = len(split)
    print(f"--> Removed {unk_size - english_size} non-english samples")
    return split


def preprocess_limericks(raw=False):
    """Preprocess the raw limericks samples."""
    if raw:
        # Read
        prep("Reading the raw poems...", key="read")
        limericks = read_limericks()
        drop(key="read")
        print(f"--> {len(limericks)} raw limerick samples read in")
        
        # Process
        prep("Processing the limericks...\n", key="process limericks")
        limericks_processed = process(limericks)
        drop(key="process limericks")
        print(f"--> {len(limericks_processed)} limerick samples processed")
        
        # Combine
        prep("Temporarily saving the poems...", key="combine")
        save_as_file(limericks_processed, f'data/processed/backup/limericks')
        drop(key="combine")
    else:
        prep("Loading previously processed datasets...", key="load")
        limericks_processed = read_as_file(f'data/processed/backup/limericks')
        
        # Remove the trailing '\n'
        handle_raw(limericks_processed)
        drop(key="load")
    print(f"--> Limerick dataset has size of {len(limericks_processed)}")
    
    return limericks_processed


def preprocess_quotes(raw=False):
    """Preprocess the raw quote samples."""
    if raw:
        prep("Loading the data...", key="load")
        quotes = read_as_file(f"data/raw/quotes.txt")
        drop(key="load")
        print(f"--> {len(quotes)} raw quote samples read in")
        
        prep("Processing the quotes...\n", key="process")
        quotes_processed = process(quotes)
        drop(key="process")
        print(f"--> {len(quotes_processed)} quote samples processed")
        
        # Temporarily saving the data
        prep("Temporarily saving the data...", key="save")
        save_as_file(quotes_processed, f'data/processed/backup/quotes')
        drop(key="save")
    else:
        prep("Loading previously processed datasets...", key="load")
        quotes_processed = read_as_file(f'data/processed/backup/quotes')
        
        # Remove the trailing '\n'
        handle_raw(quotes_processed)
        drop(key="load")
    print(f"--> Quotes dataset has size of {len(quotes_processed)}")
    
    return quotes_processed


def process_main(data, store_prefix=''):
    """Process the data."""
    prep("Pruning long segments", key="long")
    prune_long_segments(data)
    drop(key="long")
    print(f"--> Pruned combined dataset has size of {len(data)}")
    
    # Keyword extraction
    prep("Subtracting keywords...", key="keywords")
    keywords = extract_keywords(data)
    size_pre = len(keywords)  # Size before pruning out empty keywords
    assert (len(data) == len(keywords))
    
    # Prune samples with empty keywords
    prune_empty_keywords(data, keywords)
    size_post = len(keywords)  # Size after pruning out empty keywords
    assert (len(data) == len(keywords))
    drop(key="keywords")
    print(f"--> Removed {size_pre - size_post} empty keyword samples")
    
    # Split in train, test, and valid datasets
    prep("Splitting the data...", key="split")
    size = len(data)
    X_train, y_train = data[:int(size * 0.9)], keywords[:int(size * 0.9)]
    X_val, y_val = data[int(size * 0.9):int(size * 0.95)], keywords[int(size * 0.9):int(size * 0.95)]
    X_test, y_test = data[int(size * 0.95):], keywords[int(size * 0.95):]
    assert (len(X_train) == len(y_train))
    assert (len(X_val) == len(y_val))
    assert (len(X_test) == len(y_test))
    assert (len(X_train) + len(X_val) + len(X_test) == len(data))
    drop(key="split")
    print(f"--> Size of remaining samples: {size}")
    print(f"--> Training size: {len(X_train)}")
    print(f"--> Validation size: {len(X_val)}")
    print(f"--> Testing size: {len(X_test)}")
    
    # Save
    prep("Saving...", key="save")
    # Create needed folders
    make_folder(f"data/processed/full")
    make_folder(f"data/processed/train")
    make_folder(f"data/processed/valid")
    make_folder(f"data/processed/test")
    
    # Store the files
    save_as_file(data, f'data/processed/full/{store_prefix}complete')
    save_as_file(keywords, f'data/processed/full/{store_prefix}keywords')
    save_as_file(X_train, f'data/processed/train/{store_prefix}complete')
    save_as_file(y_train, f'data/processed/train/{store_prefix}keywords')
    save_as_file(X_val, f'data/processed/valid/{store_prefix}complete')
    save_as_file(y_val, f'data/processed/valid/{store_prefix}keywords')
    save_as_file(X_test, f'data/processed/test/{store_prefix}complete')
    save_as_file(y_test, f'data/processed/test/{store_prefix}keywords')
    drop(key="save")
    
    # Visualize the full dataset
    visualize(
            path=f'data/processed/full/{store_prefix}complete',
            data_tag=f"{store_prefix}data - complete",
    )


def main(raw=False,
         limericks=False,
         quotes=False):
    """
    Run the full process pipeline.
    
    :param raw: Handle raw poems by executing 'process'
    :param limericks: Process the limericks (training data)
    :param quotes: Process the quotes (pre-training data)
    """
    if limericks:
        limericks = preprocess_limericks(raw=raw)
        process_main(data=limericks)
    if quotes:
        quotes = preprocess_quotes(raw=raw)
        process_main(data=quotes, store_prefix='pre_')
    
    print("\n--> process.py has finished successfully! <--")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--process_raw", dest="raw", type=bool, default=True)  # Raw to uniform format
    parser.add_argument("--process_limericks", dest="limericks", type=bool, default=True)  # Training data
    parser.add_argument("--process_quotes", dest="quotes", type=bool, default=False)  # Pre-training data
    args = parser.parse_args()
    
    # Go back to root directory
    os.chdir("..")
    
    main(
            raw=args.raw,
            limericks=args.limericks,
            quotes=args.quotes,
    )
