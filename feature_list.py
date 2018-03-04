import re

with open('affixes.txt', 'r') as affix_file:
    affixes = set(affix_file.read().splitlines())

with open('gazetteer_names.txt', 'r') as gazeteer_file:
    gazetteer_names = set(gazeteer_file.read().splitlines())


def gazetteer_feature(sent, i):
    word = sent[i][0]
    if word.lower() in gazetteer_names:
        return 1
    else:
        return 0


def affix_feature(sent, i):
    word = sent[i][0]
    if len(word) < 4:
        return "false"
    elif len(word) >= 4:
        prefix = word[:4]
        suffix = word[-4:]
        if prefix in affixes or suffix in affixes:
            return "true"
    return "false"


def short_word_shape_feature(sent, i):
    word = sent[i][0]
    # e.g. "Corp."
    if re.match('^[A-Z][a-z]+\.$', word) is not None:
        return "Xx."

    # e.g. "INC."
    elif re.match('[A-Z]+\.', word) is not None:
        return "X."

    # e.g. "INC"
    elif re.match('^[A-Z]+$', word) is not None:
        return "X"

    # e.g. "Noah"
    elif re.match('^[A-Z][a-z]+$', word) is not None:
        return "Xx"

    # e.g. "sam"
    elif re.match('^[a-z]+$', word) is not None:
        return "x"

    # e.g. something that contains numbers
    elif re.match('\d+', word) is not None:
        return "D"
    else:
        return word


def hyphen_feature(sent, i):
    if '-' in sent[i][0]:
        return "yes_hyphen"
    else:
        return "no_hyphen"


def proper_case_feature(sent, i):
    word = sent[i][0]
    if word == word.upper():
        return "cap"
    elif word[0] == word[0].upper():
        return "prop"
    else:
        return "low"

