from collections import Counter
from nltk.corpus import conll2002

# used by the top spanish model

train_sents = list(conll2002.iob_sents('esp.train'))
occurrences = Counter()
for sent in train_sents:
    for i in range(len(sent)):
        word = sent[i][0]
        if len(word) == 4:
            occurrences.update([word])
        if len(word) > 4:
            occurrences.update([word[-4:]])
            occurrences.update([word[:4]])


final_counts = {x: occurrences[x] for x in occurrences if occurrences[x] >= 100}
with open('affixes.txt', 'w') as affixes:
    for k in final_counts.keys():
        affixes.write(k + "\n")
