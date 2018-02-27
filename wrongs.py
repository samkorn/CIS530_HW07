from nltk.corpus import conll2002
from ner import generate_features_and_labels
dev_sents = list(conll2002.iob_sents('esp.testa'))

feats, labels = generate_features_and_labels(dev_sents)

with open("results-mlp-lbfgs-200-pos1.txt", "r") as f:
    line_count = 0
    wrong_count = 0
    for line in f:
        word, gold, pred = line.split()
        if gold != pred:
            print(line + str(feats[line_count]))
            print()
            wrong_count += 1
        line_count += 1

print("\nTotal number wrong: %d" % wrong_count)

