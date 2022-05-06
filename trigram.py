import string
from collections import Counter
from typing import Dict
from functools import reduce


def preprocess_doc(doc: str):
    trans_table = str.maketrans(string.punctuation + string.digits + string.ascii_uppercase,
                                ' ' * len(string.punctuation + string.digits) + string.ascii_lowercase)
    doc = doc.translate(trans_table)
    return doc


def unigram_frequencies(doc: str) -> Counter:
    return Counter(doc.split())


def bigram_frequencies(doc: str):
    words = doc.split()
    bigrams = []
    for i in range(len(words) - 1):
        word, next_word = words[i], words[i + 1]
        bigrams.append((word, next_word))
    return Counter(bigrams)


def trigram_frequencies(doc: str):
    words = doc.split()
    trigrams = []
    for i in range(len(words) - 2):
        trigram = words[i], words[i + 1], words[i + 2]
        trigrams.append(trigram)
    return Counter(trigrams)


def bigram_probabilities(bigram_freq: Counter, unigram_freq: Counter) -> Dict[str, float]:
    bigram_probs = {}
    for bigram, freq in bigram_freq.items():
        bigram_probs[bigram] = freq / unigram_freq[bigram[0]]

    return bigram_probs


def trigram_probabilities(trigram_freq: Counter, bigram_freq: Counter) -> Dict[str, float]:
    trigram_probs = {}
    for trigram, freq in trigram_freq.items():
        first, second, _ = trigram
        trigram_probs[trigram] = freq / bigram_freq[(first, second)]
    return trigram_probs


def run_inference(test_sentence, trigram_probs):
    test_sentence = preprocess_doc(test_sentence)
    words = test_sentence.split()
    probabilities = []
    for i in range(len(words) - 2):
        word, next_word, next_to_next = words[i], words[i + 1], words[i + 2]
        probabilities.append(trigram_probs.get((word, next_word, next_to_next), 0))

    return reduce(lambda x, y: x * y, probabilities)


def main():
    doc1 = 'This is a dog. This is a cat. I love my cat. I love my dog. This is my name. My dog is cute. bucky is my dog.'
    doc1 = preprocess_doc(doc1)
    bigram_freq = bigram_frequencies(doc1)
    trigram_freq = trigram_frequencies(doc1)
    bigram_probs = trigram_probabilities(trigram_freq, bigram_freq)
    test_sentence = 'this is a dog'
    print(run_inference(test_sentence, bigram_probs))


if __name__ == '__main__':
    main()
