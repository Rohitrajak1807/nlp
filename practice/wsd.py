from nltk.corpus import wordnet, stopwords
from nltk.tokenize import word_tokenize
import string


def preprocess_doc(doc: str):
    trans_table = str.maketrans(string.punctuation + string.digits + string.ascii_uppercase,
                                ' ' * len(string.punctuation + string.digits) + string.ascii_lowercase)
    doc = doc.translate(trans_table)
    return doc


def overlap_context(synset, sentence):
    gloss = set(word_tokenize(synset.definition()))
    for example in synset.examples():
        gloss.union(word_tokenize(example))
    gloss = gloss.difference(stopwords.words('english'))
    sentence = set(sentence.split())
    sentence = sentence.difference(stopwords.words('english'))
    return len(gloss.intersection(sentence))


def lesk(word, sentence):
    word = preprocess_doc(word)
    sentence = preprocess_doc(sentence)
    best_sense = None
    max_overlap = 0
    word = wordnet.morphy(word) if wordnet.morphy(word) is not None else word

    for sense in wordnet.synsets(word):
        overlap = overlap_context(sense, sentence)
        for h in sense.hyponyms():
            overlap += overlap_context(h, sentence)
        if overlap > max_overlap:
            max_overlap = overlap
            best_sense = sense
    return best_sense


def main():
    sentence = input("Enter the Sentence (or) Context :")
    word = input("Enter the word :")

    a = lesk(word, sentence)
    print("\n\nSynset:", a)
    if a is not None:
        print("Meaning:", a.definition())
        num = 0
        print("\nExamples:")
        for i in a.examples():
            num = num + 1
            print(str(num) + '.' + ')', i)


if __name__ == '__main__':
    main()

# functionwords = ['about', 'across', 'against', 'along', 'around', 'at',
#                  'behind', 'beside', 'besides', 'by', 'despite', 'down',
#                  'during', 'for', 'from', 'in', 'inside', 'into', 'near', 'of',
#                  'off', 'on', 'onto', 'over', 'through', 'to', 'toward',
#                  'with', 'within', 'without', 'anything', 'everything',
#                  'anyone', 'everyone', 'ones', 'such', 'it', 'itself',
#                  'something', 'nothing', 'someone', 'the', 'some', 'this',
#                  'that', 'every', 'all', 'both', 'one', 'first', 'other',
#                  'next', 'many', 'much', 'more', 'most', 'several', 'no', 'a',
#                  'an', 'any', 'each', 'no', 'half', 'twice', 'two', 'second',
#                  'another', 'last', 'few', 'little', 'less', 'least', 'own',
#                  'and', 'but', 'after', 'when', 'as', 'because', 'if', 'what',
#                  'where', 'which', 'how', 'than', 'or', 'so', 'before', 'since',
#                  'while', 'although', 'though', 'who', 'whose', 'can', 'may',
#                  'will', 'shall', 'could', 'be', 'do', 'have', 'might', 'would',
#                  'should', 'must', 'here', 'there', 'now', 'then', 'always',
#                  'never', 'sometimes', 'usually', 'often', 'therefore',
#                  'however', 'besides', 'moreover', 'though', 'otherwise',
#                  'else', 'instead', 'anyway', 'incidentally', 'meanwhile']
