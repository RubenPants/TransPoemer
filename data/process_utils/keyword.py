"""
keyword.py

Processing files related to the keywords.
"""
import re

from rake_nltk import Rake


def extract_keywords(poems):
    """
    Let only the keywords remain in each data segment. To extract the keywords, NLTK's RAKE (Rapid Automatic Keyword
    Extraction) algorithm is used.

    :return: List of strings where only the keywords remain
    """
    rake = Rake(max_length=1)
    result = []
    for d in poems:
        # Extract the keywords
        rake.extract_keywords_from_text(d)
        
        # Extract all the keywords
        keyword_list = rake.get_ranked_phrases()
        
        # Remove keywords that contain less than 3 characters
        keyword_list = [k for k in keyword_list if len(k) >= 3]
        
        # Remove keywords incorporated in other keywords
        remove_duplicate_keywords(keyword_list)
        
        # Enlarge the model's creativity by only using the 10 highest ranked (most important) keywords
        keyword_list = keyword_list[:10]
        
        # Get starting positions of each of the keywords
        keyword_dict = dict()
        for k in keyword_list:
            try:
                # Keywords only exist out of letter, numbers, spaces, apostrophes, or slashes
                if re.match("^[a-zA-Z0-9\s\w'\w-]*$", k) and len(k) > 3:  # Keywords must be at least of size 4
                    search = re.compile(k)
                    for m in search.finditer(d):
                        keyword_dict[m.start()] = m.group()  # Unique starting position
            except re.error:
                pass  # Ignore keywords that raise an exception (e.g. contain parenthesis), very seldom
        
        # Sort the positions (together with corresponding keywords) in ascending order
        keywords_sorted = sorted(keyword_dict.items())
        result.append(" ".join(map(lambda x: x[1], keywords_sorted)))  # Sentence containing only the keywords
    return result


def remove_duplicate_keywords(keywords):
    """ Remove duplicates in the keyword-list. """
    remove_indices = []  # Indices of data-segments that must be removed
    for k_index, k in enumerate(keywords):
        if len([k for segment in keywords if k in segment]) > 1:
            remove_indices.append(k_index)
    
    # Remove bad samples
    for i in remove_indices[::-1]:
        del keywords[i]


def prune_empty_keywords(poems, keyword_list):
    """ Prune the poems that have an empty keyword-list. """
    remove_indices = []
    for index, keyword in enumerate(keyword_list):
        if len(keyword) == 0:
            remove_indices.append(index)
    
    # Remove bad samples
    for i in remove_indices[::-1]:
        del poems[i]
        del keyword_list[i]
