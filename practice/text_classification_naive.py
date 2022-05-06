import pickle
import random
import nltk
from nltk.corpus import movie_reviews


def find_features(review, word_features):
    words = set(review)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features


def main():
    random.seed(1)
    documents = []
    for category in movie_reviews.categories():
        for fileid in movie_reviews.fileids(category):
            documents.append(
                (list(movie_reviews.words(fileid)), category)
            )
    random.shuffle(documents)
    all_words = []
    for w in movie_reviews.words():
        all_words.append(w.lower())

    all_words = nltk.FreqDist(all_words)
    word_features = list(all_words.keys())[:5000]
    feature_sets = [
        (find_features(rev, word_features), category) for (rev, category) in documents
    ]
    training_set = feature_sets[:1900]
    testing_set = feature_sets[1900:]
    classifer = nltk.NaiveBayesClassifier.train(training_set)
    print(f'Accuracy: {nltk.classify.accuracy(classifer, testing_set) * 100}')
    classifer.show_most_informative_features()
    with open('navieb.pickle', 'wb') as f:
        pickle.dump(classifer, f)

    print(classifer.classify(find_features('I hate the movie', word_features)))


if __name__ == '__main__':
    main()
