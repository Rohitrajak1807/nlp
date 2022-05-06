import string
import sys
from collections import OrderedDict
from typing import List


def read_file(file: str) -> str:
    try:
        with open(file, 'r') as f:
            data = f.read()
        return data
    except IOError as e:
        print(f'error: {e}')
        sys.exit(e)


def get_words(text: str, trans_table=None) -> List[str]:
    if trans_table is None:
        trans_table = str.maketrans(string.punctuation + string.ascii_uppercase,
                                    ' ' * len(string.punctuation) + string.ascii_lowercase)
    text = text.translate(trans_table)
    word_list = text.split()
    return word_list


def count_unigram_frequency(word_list: List[str]) -> OrderedDict[str, int]:
    frequencies = OrderedDict()
    for word in word_list:
        if word in frequencies:
            frequencies[word] += 1
        else:
            frequencies[word] = 1
    return frequencies
