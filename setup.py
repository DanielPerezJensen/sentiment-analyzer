import nltk
import random
import pickle
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from nltk.tokenize import word_tokenize
from nltk.classify.scikitlearn import SklearnClassifier


def find_features(document, word_features):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features


def main():

    short_pos = open("reviews/positive.txt", "r").read()
    short_neg = open("reviews/negative.txt", "r").read()

    # move this up here
    all_words = []
    documents = []

    allowed_word_types = ["J"]

    # add all positive and negative reviews to documents-list with label
    for review in short_pos.split('\n'):
        documents.append((review, "pos"))
        words = word_tokenize(review)
        pos = nltk.pos_tag(words)
        for w in pos:
            if w[1][0] in allowed_word_types:
                all_words.append(w[0].lower())
    for review in short_neg.split('\n'):
        documents.append((review, "neg"))
        words = word_tokenize(review)
        pos = nltk.pos_tag(words)
        for w in pos:
            if w[1][0] in allowed_word_types:
                all_words.append(w[0].lower())


    all_words = nltk.FreqDist(all_words)

    # use 5000 most used words as features
    word_features = list(all_words.keys())[:5000]

    # loop through all data_types to save as a .pickle
    data_types = [word_features, documents, all_words]
    data_types_string = ["word_features", "documents", "all_words"]
    for i in range(len(data_types)):
        string = data_types_string[i] + ".pickle"
        with open("pickled/data/" + string, "wb") as write_data:
            pickle.dump(data_types[i], write_data)


    # training and testing data added to feature sets
    featuresets = [(find_features(review, word_features), category) for (review, category) in documents]
    random.shuffle(featuresets)

    testing_set = featuresets[10000:]
    training_set = featuresets[:10000]

    print("Training: NB_classifier")
    NB_classifier = nltk.NaiveBayesClassifier.train(training_set)
    print("Done training: NB_classifier")
    MNB_classifier = SklearnClassifier(MultinomialNB())
    BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
    LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
    LinearSVC_classifier = SklearnClassifier(LinearSVC())

    classifiers = [NB_classifier, 
                   MNB_classifier, 
                   BernoulliNB_classifier, 
                   LogisticRegression_classifier, 
                   LinearSVC_classifier]
    
    classifier_strings = ["NB_classifier",
                          "MNB_classifier",
                          "BernoulliNB_classifier",
                          "LogisticRegression_classifier",
                          "LinearSVC_classifier"]
    
    # loop through all classifiers and save it as a .pickle
    for i in range(len(classifiers)):
        
        if classifier_strings[i] is not "NB_classifier":
            print("Training: " + classifier_strings[i])
            classifiers[i].train(training_set)
            print("Done training: " + classifier_strings[i])
        string = classifier_strings[i] + ".pickle"

        print("Pickling: " + string)
        with open("pickled/algorithms/" + string, "wb") as write_data:
            pickle.dump(classifiers[i], write_data)
        print("Done Pickling: " + string)

if __name__ == "__main__":
    main()