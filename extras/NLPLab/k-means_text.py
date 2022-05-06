from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer


if __name__ == '__main__':
    documents = ["This little kitty came to play when I was eating at a restaurant.",
                 "Merley has the best squooshy kitten belly.",
                 "Google Translate app is incredible.",
                 "If you open 100 tab in google you get a smiley face.",
                 "Best cat photo I've ever taken.",
                 "Climbing ninja cat.",
                 "Impressed with google map feedback.",
                 "Key promoter extension for Google Chrome."]

    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(documents)

    true_k = 2
    # model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
    model = KMeans(n_clusters=true_k)
    model.fit(X)
    print("Top terms per cluster:")
    order_centroids = model.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names_out()
    for i in range(true_k):
        print("Cluster %d:" % i),
        for ind in order_centroids[i, :10]:
            print(' %s' % terms[ind])
    print("\n")
    print("Prediction")

    Y = vectorizer.transform(["chrome browser to open."])
    prediction = model.predict(Y)
    print(prediction)

    Y = vectorizer.transform(["My cat is hungry."])
    prediction = model.predict(Y)
    print(prediction)
