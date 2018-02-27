from nltk.corpus import conll2002
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Perceptron
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import re
from collections import Set
import feature_list as fl

# Assignment 7: NER
# This is just to help you get going. Feel free to
# add to or modify any part of it.

def word2features(sent, i):
    """ The function generates all features
    for the word at position i in the
    sentence."""
    features = []

    # word before
    # should we do something like <START>?
    if i > 0:
        features.append(('-1word', sent[i - 1][0]))
        features.append(('-1affix', fl.affix_feautre(sent, i - 1)))
        features.append(('-1short_shape', fl.short_word_shape_feature(sent, i - 1)))
        features.append(('-1gazetteer', fl.gazetteer_feature(sent, i - 1)))

    # word itself
    features.append(('0word', sent[i][0]))

    # word after
    if i < len(sent) - 1:
        features.append(('+1hyphen', fl.hyphen_feature(sent, i + 1)))
        features.append(('+1word', sent[i + 1][0]))
        features.append(('affix', fl.affix_feautre(sent, i + 1)))
        features.append(('+1short_shape', fl.short_word_shape_feature(sent, i + 1)))
        features.append(('+1gazetteer', fl.gazetteer_feature(sent, i + 1)))

    # part of speech (0)
    features.append(('0pos', sent[i][1]))

    # # part of speech (-1)
    # features.append(('-1pos', sent[i-1][1]))

    # # part of speech (-1 & 0)
    # features.append(('-1pos&0pos', sent[i-1][1]+"&"+sent[i][1]))

    # proper case
    features.append(('prop', fl.proper_case_feature(sent, i)))

    # presence of hyphen
    features.append(('hyphen', fl.hyphen_feature(sent, i)))

    # gets the word's short shape
    features.append(('short_shape', fl.short_word_shape_feature(sent, i)))

    # if a popular affix in the training set is within the word
    features.append(('affix', fl.affix_feautre(sent, i)))

    # if the word is in the gazetteer set
    features.append(('gazetteer', fl.gazetteer_feature(sent, i)))

    return dict(features)


def generate_features_and_labels(sentences):
    feats = []
    labels = []

    for sent in sentences:
        for i in range(len(sent)):
            feats.append(word2features(sent, i))
            labels.append(sent[i][-1])

    return feats, labels


def generate_tensor(vectorizer, features, training=False):
    if training:
        tensor = vectorizer.fit_transform(features)
    else:
        tensor = vectorizer.transform(features)

    return tensor


if __name__ == "__main__":
    # Load the training data
    train_sents = list(conll2002.iob_sents('esp.train'))
    dev_sents = list(conll2002.iob_sents('esp.testa'))
    test_sents = list(conll2002.iob_sents('esp.testb'))

    # Generate training tensor
    vectorizer = DictVectorizer()
    train_feats, train_labels = generate_features_and_labels(train_sents)
    X_train = generate_tensor(vectorizer, train_feats, training=True)

    # Train the model
    # model = Perceptron(verbose=True, max_iter=20)
    model = MLPClassifier(verbose=True, solver='lbfgs', max_iter=200)
    model.fit(X_train, train_labels)

    # Generate test tensor (switch to test_sents for submission output)
    test_feats, test_labels = generate_features_and_labels(dev_sents)
    X_test = generate_tensor(vectorizer, test_feats)

    # Generate labels using trained model
    y_pred = model.predict(X_test)

    j = 0
    print("Writing to results.txt")
    # format is: word gold pred
    with open("results.txt", "w") as out:
        for sent in dev_sents:
            for i in range(len(sent)):
                word = sent[i][0]
                gold = sent[i][-1]
                pred = y_pred[j]
                j += 1
                out.write("{}\t{}\t{}\n".format(word, gold, pred))
        out.write("\n")

    print("Now run: python conlleval.py results.txt")
