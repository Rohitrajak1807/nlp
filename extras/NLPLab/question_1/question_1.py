# 1. Write a program to measure the document similarity.

import math
import sys
from typing import List
from collections import OrderedDict
# from utilities import read_file
# from utilities import get_words
# from utilities import count_unigram_frequency

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


def get_word_frequency_from_file(file: str) -> OrderedDict[str, int]:
    text = read_file(file)
    words = get_words(text)
    frequencies = count_unigram_frequency(words)
    return frequencies


def dot_prod(v1: List[int], v2: List[int]) -> float:
    total = 0.0
    for a, b in zip(v1, v2):
        total += (a * b)
    return total


def vector_angle(v1: List[int], v2: List[int]) -> float:
    numerator = dot_prod(v1, v2)
    denominator = math.sqrt(dot_prod(v1, v1) * dot_prod(v2, v2))
    return math.acos(numerator / denominator)


def get_document_similarity(file_1: str, file_2: str) -> float:
    words_1 = get_word_frequency_from_file(file_1)
    words_2 = get_word_frequency_from_file(file_2)
    v1 = list(words_1.values())
    v2 = list(words_2.values())
    dist = vector_angle(v1, v2)
    return dist


if __name__ == '__main__':
    angle = get_document_similarity(sys.argv[1], sys.argv[2])
    print(f'Similarity = {angle} radians')
