# 2. Write a program to build a bigram language model.
import string
from typing import List, Dict, Tuple

# from utilities import read_file, count_unigram_frequency, get_words

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


translation_table = str.maketrans('', '', string.punctuation)


def to_lowercase(text: str):
    trans_table = str.maketrans(string.ascii_uppercase, string.ascii_lowercase)
    return text.translate(trans_table)


def count_bigram_frequency(word_list: List[str]):
    bigram_freq = {}
    for i in range(len(word_list) - 1):
        if (word_list[i], word_list[i + 1]) in bigram_freq:
            bigram_freq[(word_list[i], word_list[i + 1])] += 1
        else:
            bigram_freq[(word_list[i], word_list[i + 1])] = 1
    return bigram_freq


def get_bigram_probabilities(unigram_freqs: Dict[str, int], bigram_freqs: Dict[Tuple[str, str], int]) -> Dict:
    probabilities = {}
    for bigram, freq in bigram_freqs.items():
        word_1, _ = bigram
        probabilities[bigram] = freq / unigram_freqs[word_1]
    return probabilities


def prepare_text(text: str) -> List[str]:
    words = get_words(text, translation_table)
    words = ' '.join(words)
    words = to_lowercase(words)
    return words.split()


def run_inference(text: str, bigram_probabilities: Dict) -> float:
    words = get_words(text, translation_table)
    words = ' '.join(words)
    words = to_lowercase(words)
    words = words.split()
    bigrams = []
    out: float = 1
    for i in range(len(words) - 1):
        bigrams.append((words[i], words[i + 1]))

    for bigram in bigrams:
        if bigram in bigram_probabilities:
            out *= bigram_probabilities[bigram]
        else:
            out *= 0
    return out


def main():
    text = read_file('text.txt')
    words = prepare_text(text)
    unigram_freqs = count_unigram_frequency(words)
    bigram_freqs = count_bigram_frequency(words)
    probabilities = get_bigram_probabilities(unigram_freqs, bigram_freqs)
    print(probabilities)
    sample_text = 'This is my dog.'
    print(run_inference(sample_text, probabilities))


if __name__ == '__main__':
    main()
