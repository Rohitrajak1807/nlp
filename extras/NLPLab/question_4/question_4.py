# 4. Write a program to find the word Collocations.
from nltk import word_tokenize
from nltk.corpus import state_union
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures


def main():
    corpus = state_union.raw('2006-GWBush.txt')
    sentences = word_tokenize(corpus)
    bigram_collocations = BigramCollocationFinder.from_words(sentences)
    print(bigram_collocations.nbest(BigramAssocMeasures.likelihood_ratio, 10))


if __name__ == '__main__':
    main()
