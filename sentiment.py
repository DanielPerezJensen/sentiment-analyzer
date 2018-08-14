import pickle
from nltk.classify import ClassifierI
from nltk.tokenize import word_tokenize
from statistics import mode


class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for classifier in self._classifiers:
            vote = classifier.classify(features)
            votes.append(vote)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for classifier in self._classifiers:
            vote = classifier.classify(features)
            votes.append(vote)

        choice_votes = votes.count(mode(votes))
        confidence = choice_votes / len(votes)
        return confidence


def find_features(document, word_features):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features


def sentiment(text):

    classifier_strings = ["NB_classifier",
                          "MNB_classifier",
                          "BernoulliNB_classifier",
                          "LogisticRegression_classifier",
                          "LinearSVC_classifier"]

    classifiers = []
    for classifier_string in classifier_strings:
        string = "pickled/algorithms/" + classifier_string + ".pickle"
        with open(string, "rb") as classifier_file:
            classifier = pickle.load(classifier_file)
            classifiers.append(classifier)

    NB_classifier = classifiers[0]
    MNB_classifier = classifiers[1]
    BernoulliNB_classifier = classifiers[2]
    LogisticRegression_classifier = classifiers[3]
    LinearSVC_classifier = classifiers[4]

    voted_clf = VoteClassifier(NB_classifier,
                               MNB_classifier,
                               BernoulliNB_classifier,
                               LogisticRegression_classifier,
                               LinearSVC_classifier)

    with open("pickled/data/word_features.pickle", "rb") as word_features_file:
        word_features = pickle.load(word_features_file)
        features = find_features(text, word_features)

    return voted_clf.classify(features), voted_clf.confidence(features)
