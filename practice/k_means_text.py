from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

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
    features = vectorizer.fit_transform(documents)
    k = 2

    model = KMeans(n_clusters=2)
    model.fit(features)

    y = vectorizer.transform(['open a tab in chrome'])
    prediction = model.predict(y)
    print(prediction)

    y = vectorizer.transform(['my cat is hungry'])
    prediction = model.predict(y)
    print(prediction)
