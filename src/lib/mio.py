from __future__ import print_function
import codecs
import numpy as np
import sys

def load_embeddings_file(file_name, sep=" ",lower=False):
    """
    load embeddings file
    """
    emb={}
    for line in open(file_name, errors='ignore', encoding='utf-8'):
        try:
            fields = line.strip().split(sep)
            vec = [float(x) for x in fields[1:]]
            word = fields[0]
            if lower:
                word = word.lower()
            emb[word] = vec
        except ValueError:
            print("Error converting: {}".format(line))

    print("loaded pre-trained embeddings (word->emb_vec) size: {} (lower: {})".format(len(emb.keys()), lower))
    return emb, len(emb[word])

def read_conll_file(file_name, raw=False):
    """
    read in conll file
    word1    tag1
    ...      ...
    wordN    tagN

    Sentences MUST be separated by newlines!

    :param file_name: file to read in
    :param raw: if raw text file (with one sentence per line) -- adds 'DUMMY' label
    :return: generator of instances ((list of  words, list of tags) pairs)

    """
    current_words = []
    current_tags = []
    
    for line in codecs.open(file_name, encoding='utf-8'):
        #line = line.strip()
        line = line[:-1]

        if line:
            if raw:
                current_words = line.split() ## simple splitting by space
                current_tags = ['DUMMY' for _ in current_words]
                yield (current_words, current_tags)

            else:
                if len(line.split("\t")) != 2:
                    if len(line.split("\t")) == 1: # emtpy words in gimpel
                        raise IOError("Issue with input file - doesn't have a tag or token?")
                    else:
                        print("erroneous line: {} (line number: {}) ".format(line), file=sys.stderr)
                        exit()
                else:
                    word, tag = line.split('\t')
                current_words.append(word)
                current_tags.append(tag)

        else:
            if current_words and not raw: #skip emtpy lines
                yield (current_words, current_tags)
            current_words = []
            current_tags = []

    # check for last one
    if current_tags != [] and not raw:
        yield (current_words, current_tags)

    
if __name__=="__main__":
    allsents=[]
    unique_tokens=set()
    unique_tokens_lower=set()
    for words, tags in read_conll_file("data/da-ud-train.conllu"):
        allsents.append(words)
        unique_tokens.update(words)
        unique_tokens_lower.update([w.lower() for w in words])
    assert(len(allsents)==4868)
    assert(len(unique_tokens)==17552)

