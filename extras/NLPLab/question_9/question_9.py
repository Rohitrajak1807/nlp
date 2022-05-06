# 9. Build a simple Text Classification Program.
import nltk
import random
from nltk.corpus import movie_reviews
import pickle


def find_features(document, word_features):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features


def main():
    documents = [(list(movie_reviews.words(fileid)), category)
                 for category in movie_reviews.categories()
                 for fileid in movie_reviews.fileids(category)]
    random.shuffle(documents)
    # print(documents[1])
    all_words = []
    for w in movie_reviews.words():
        all_words.append(w.lower())

    all_words = nltk.FreqDist(all_words)
    print(all_words.most_common(15))
    word_features = list(all_words.keys())[:3000]
    # print(find_features(movie_reviews.words('neg/cv000_29416.txt'), word_features))
    featuresets = [(find_features(rev, word_features), category) for (rev, category) in documents]
    training_set = featuresets[:1900]
    testing_set = featuresets[1900:]
    classifier = nltk.NaiveBayesClassifier.train(training_set)
    print(f'Accuracy: {nltk.classify.accuracy(classifier, testing_set) * 100}')
    classifier.show_most_informative_features(15)
    with open('naiveb.pickle', 'wb') as f:
        pickle.dump(classifier, f)


if __name__ == '__main__':
    main()
