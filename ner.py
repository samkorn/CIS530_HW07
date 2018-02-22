from nltk.corpus import conll2002
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Perceptron
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_recall_fscore_support
import numpy as np

# Assignment 7: NER
# This is just to help you get going. Feel free to
# add to or modify any part of it.


def get_words_in_window(word_tuple, o):
    """ This takes the word in question and
    the offset with respect to the instance
    word """
    o = str(o)
    features = [(o + 'word', word_tuple[0])]
    return features


def is_proper_case(word):
    if word[0] == word[0].upper():
        return 1
    else:
        return 0


def word2features(sent, i):
    """ The function generates all features
    for the word at position i in the
    sentence."""
    features = []

    # the [-1,+1] window of words around the token
    for o in [-1,0,1]:
        if i+o >= 0 and i+o < len(sent):
            word_tuple = sent[i+o]
            word_window = get_words_in_window(word_tuple, o)
            features.extend(word_window)

    # part of speech
    pos = ('pos', sent[i][1])
    features.append(pos)

    # prop = ('prop', is_proper_case(sent[i][0]))
    # features.append(prop)

    return dict(features)


def generate_features_and_labels(sentences):
    feats = []
    labels = []

    for sent in sentences:
        for i in range(len(sent)):
            sent_feats = word2features(sent,i)
            feats.append(sent_feats)
            labels.append(sent[i][-1])

    return feats, labels


def generate_tensor(vectorizer, sentences, fit=False):
    feats, labels = generate_features_and_labels(sentences)

    if fit:
        tensor = vectorizer.fit_transform(feats)
    else:
        tensor = vectorizer.transform(feats)

    return tensor, labels


if __name__ == "__main__":
    # Load the training data
    train_sents = list(conll2002.iob_sents('esp.train'))
    dev_sents = list(conll2002.iob_sents('esp.testa'))
    test_sents = list(conll2002.iob_sents('esp.testb'))

    # Generate training tensor
    vectorizer = DictVectorizer()
    X_train, train_labels = generate_tensor(vectorizer, train_sents, fit=True)

    # Train the model
    # model = Perceptron(verbose=True, max_iter=20)
    model = MLPClassifier(verbose=True, solver='lbfgs', max_iter=200)
    model.fit(X_train, train_labels)

    # Generate test tensor (switch to test_sents for submission output)
    X_test, test_labels = generate_tensor(vectorizer, dev_sents)

    # Generate labels using trained model
    y_pred = model.predict(X_test)


    # wrongs = []
    j = 0
    print("Writing to results.txt")
    # format is: word gold pred
    with open("results.txt", "w") as out:
        for sent in dev_sents:
            for i in range(len(sent)):
                word = sent[i][0]
                gold = sent[i][-1]
                pred = y_pred[j]
                # if gold != pred:
                #
                # j += 1
                out.write("{}\t{}\t{}\n".format(word,gold,pred))
        out.write("\n")

    print("Now run: python conlleval.py results.txt")









