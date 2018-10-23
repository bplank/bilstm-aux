import codecs
import numpy as np
from scipy import linalg
from collections import defaultdict
import re

class Seq(object):
    """
    Seq (sequential data) object
    """
    __slots__ = ['words', 'tags', 'task_id', 'pred_tags', 'tag_confidences']

    def __init__(self, words, tags=None, task_id=None):
        self.words = words
        self.tags = tags
        self.task_id = task_id
        self.tag_confidences = []
        self.pred_tags = []

    def evaluate(self):
        correct = np.sum([i == j for i, j in zip(self.pred_tags, self.tags)])
        total = len(self.tags)
        return correct, total

class SeqData(object):
    """
    Reads in all data files and maps them to task ids
    """
    __slots__ = ['seqs', 'task_ids']

    def __init__(self, list_folders_name):
        self.seqs = []
        self.task_ids = set()
        for i, file_name in enumerate(list_folders_name):
            task_id = "task{}".format(i)
            self.task_ids.add(task_id)
            for word_seq, tag_seq in read_conll_file(file_name):
                self.seqs.append(Seq(word_seq, tag_seq, task_id))

    def __iter__(self):
        """iterate over data"""
        for seq in self.seqs:
            yield seq

def load_dict(file_name):
    d = defaultdict(set)
    dict_values = set()
    for line in codecs.open(file_name,encoding="utf-8", errors="ignore"):
        word, tag = line.strip().split("\t")
        d[word].add(tag)
        dict_values.add(tag)
    print("Loaded dictionary with {} word types".format(len(d)))
    return d, sorted(dict_values)

def load_embeddings_file(file_name, sep=" ",lower=False, normalize=False):
    """
    load embeddings file
    """
    emb={}
    for line in open(file_name, errors='ignore', encoding='utf-8'):
        try:
            fields = re.split(" ", line)
            if fields[-1] == "\n":
                fields = fields[:-1] 
            vec = [float(x) for x in fields[1:]]
            word = fields[0]
            if lower:
                word = word.lower()
            emb[word] = vec
            if normalize:
                emb[word] /= linalg.norm(emb[w])
        except ValueError:
            print("Error converting: {}".format(line))

    print("loaded pre-trained embeddings (word->emb_vec) size: {}".format(len(emb)))
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
                        raise IOError("erroneous line: {}".format(line))
                else:
                    word, tag = line.split('\t')
                    if not tag:
                        raise IOError("empty tag in line line: {}".format(line))
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

