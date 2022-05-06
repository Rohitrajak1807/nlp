import string
from nltk.tokenize import word_tokenize
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures


def preprocess_doc(doc: str):
    trans_table = str.maketrans(string.punctuation + string.digits + string.ascii_uppercase,
                                ' ' * len(string.punctuation + string.digits) + string.ascii_lowercase)
    doc = doc.translate(trans_table)
    return doc


def main():
    doc = '''Only yesterday, we laid to rest the mortal remains of our beloved President, Franklin Delano Roosevelt. 
    At a time like this, words are inadequate. The most eloquent tribute would be a reverent silence. Yet, 
    in this decisive hour, when world events are moving so rapidly, our silence might be misunderstood and might give 
    comfort to our enemies. In His infinite wisdom, Almighty God has seen fit to take from us a great man who loved, 
    and was beloved by, all humanity. No man could possibly fill the tremendous void left by the passing of that 
    noble soul. No words can ease the aching hearts of untold millions of every race, creed and color. The world 
    knows it has lost a heroic champion of justice and freedom. Tragic fate has thrust upon us grave 
    responsibilities. We must carry on.Our departed leader never looked backward. He looked forward and moved 
    forward. That is what he would want us to do. That is what America will do. So much blood has already been shed 
    for the ideals which we cherish, and for which Franklin Delano Roosevelt lived and died, that we dare not permit 
    even a momentary pause in the hard fight for victory. Today, the entire world is looking to America for 
    enlightened leadership to peace and progress. Such a leadership requires vision, courage and tolerance. It can be 
    provided only by a united nation deeply devoted to the highest ideals. With great humility I call upon all 
    Americans to help me keep our nation united in defense of those ideals which have been so eloquently proclaimed 
    by Franklin Roosevelt. I want in turn to assure my fellow Americans and all of those who love peace and liberty 
    throughout the world that I will support and defend those ideals with all my strength and all my heart. That is 
    my duty and I shall not shirk it. So that there can be no possible misunderstanding, both Germany and Japan can 
    be certain, beyond any shadow of a doubt, that America will continue the fight for freedom until no vestige of 
    resistance remains! '''
    doc = preprocess_doc(doc)
    tokens = word_tokenize(doc)
    finder = BigramCollocationFinder.from_words(tokens)
    print(finder.nbest(BigramAssocMeasures.likelihood_ratio, 10))


if __name__ == '__main__':
    main()
