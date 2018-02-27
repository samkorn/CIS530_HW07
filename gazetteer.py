import re

with open('gazetteer_data/stopwords.txt', 'r') as stop_file:
    stopwords = set(stop_file.read().splitlines())


def get_gazetteer_names(filename):
    names = []
    with open(filename, 'r') as f:
        for l in f.readlines():
            l = l.split('\t')
            n = l[1].lower().split()
            for sub_word in n:
                if re.match(r'.*[\W]+.*', sub_word) is None and sub_word not in stopwords:
                    names.append(sub_word)
    return set(names)

names = get_gazetteer_names('gazetteer_data/MX.txt')
with open('gazetteer_data/gazetteer_names.txt', 'w') as out:
    for n in names:
        out.write(n + '\n')
    out.close()
