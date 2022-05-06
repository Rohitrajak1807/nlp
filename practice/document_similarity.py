import math
import string
from typing import List, Dict
from collections import Counter


def preprocess_tokenize(doc: str):
    trans_table = str.maketrans(string.punctuation + string.digits + string.ascii_uppercase,
                                ' ' * len(string.punctuation + string.digits) + string.ascii_lowercase)
    doc = doc.lower()
    doc = doc.translate(trans_table)
    return doc


def get_vocabulary_mapping(docs: List[str]) -> dict:
    all_tokens = [word for doc in docs for word in doc.split()]
    vocabulary = list(set(all_tokens))
    mapping = {}
    for idx, word in enumerate(vocabulary):
        mapping[word] = idx

    return mapping


def get_unigram_freqs(doc: str):
    all_tokens = [word for word in doc.split()]
    return Counter(all_tokens)


def to_vector(doc: str, mapping: Dict[str, int]):
    frequencies = get_unigram_freqs(doc)
    doc_vector = [0] * len(mapping)
    for word in doc.split():
        index = mapping[word]
        frequency = frequencies[word]
        doc_vector[index] = frequency
    return doc_vector


def dot_prod(v1, v2):
    total = 0
    for a, b in zip(v1, v2):
        total += a * b
    return total


def vector_magnitude(v):
    return math.sqrt(dot_prod(v, v))


def main():
    doc1 = "I want to break free....."
    doc2 = "I want to break free from your lies!"
    doc1 = preprocess_tokenize(doc1)
    doc2 = preprocess_tokenize(doc2)
    mapping = get_vocabulary_mapping([doc1, doc2])
    doc1_vector = to_vector(doc1, mapping)
    doc2_vector = to_vector(doc2, mapping)
    cosine = dot_prod(doc1_vector, doc2_vector) / (vector_magnitude(doc1_vector) * vector_magnitude(doc2_vector))
    print(cosine)


if __name__ == '__main__':
    main()
