import codecs
import numpy as np
from scipy import linalg
from collections import defaultdict
import re
import sys

class Seq(object):
    """
    Seq (sequential data) object
    """
    __slots__ = ['words', 'tags', 'task_id', 'pred_tags', 'tag_confidences', 'embeds']

    def __init__(self, words, tags=None, task_id=None, embeds=None):
        self.words = words
        self.tags = tags
        self.task_id = task_id
        self.embeds = embeds
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

    def __init__(self, list_folders_name, raw=False, embeds_in_file=False):
        self.seqs = []
        self.task_ids = set()
        for i, file_name in enumerate(list_folders_name):
            task_id = "task{}".format(i)
            self.task_ids.add(task_id)
            for word_seq, tag_seq, emb_seq in read_conll_file(file_name, raw=raw, embeds_in_file=embeds_in_file):
                self.seqs.append(Seq(word_seq, tag_seq, task_id, embeds=emb_seq))
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
    first = True
    for line in open(file_name, errors='ignore', encoding='utf-8'):
        try:
            fields = re.split(" ", line)
            if len(fields) < 5 and first:
                first = False
                continue
            if fields[-1] == "\n":
                fields = fields[:-1] 
            vec = [float(x) for x in fields[1:]]
            word = fields[0]
            if lower:
                word = word.lower()
            emb[word] = vec
            if normalize:
                emb[word] /= linalg.norm(emb[w])
            first = False
        except ValueError:
            print("Error converting: {}".format(line))

    print("loaded pre-trained embeddings (word->emb_vec) size: {}".format(len(emb)))
    return emb, len(emb[word])

def read_conll_file(file_name, raw=False, embeds_in_file=False):
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
    if raw and embeds_in_file:
        print("sorry, combining --raw and --embeds_in_file is currently not supported, please retry with adding dummy labels")
        exit(1)
    current_words = []
    current_tags = []
    current_embeds = []
    ws_pattern = re.compile("^\s+$") # match emtpy lines that contain some whitespace
    
    for line in codecs.open(file_name, encoding='utf-8'):
        #line = line.strip()
        line = line[:-1]

        if not line or ws_pattern.match(line):
            if current_words and not raw: #skip emtpy lines
                yield (current_words, current_tags, current_embeds)
            current_words = []
            current_tags = []
            current_embeds = []
        else:
            if raw:
                current_words = line.split() ## simple splitting by whitespace
                current_tags = ['DUMMY' for _ in current_words]
                current_embeds = [[] for _ in current_words]
                yield (current_words, current_tags, current_embeds)
            else:
                tok = line.split('\t')
                if len(tok) == 2 and not embeds_in_file:
                    word, tag = tok
                    embed = []
                    if not tag:
                        raise IOError("empty tag in line line: {}".format(line))
                elif len(tok) == 3 and embeds_in_file:
                    word = tok[0]
                    tag = tok[1]
                    embed = [float(x) for x in tok[2][4:].split(',')]
                else:
                    if len(line.split("\t")) == 1: # emtpy words in gimpel
                        raise IOError("Issue with input file - doesn't have a tag or token?")
                    else:
                        raise IOError("erroneous line: {}".format(line))
                current_words.append(word)
                current_tags.append(tag)
                current_embeds.append(embed)
    # check for last one
    if current_tags != [] and not raw:
        yield (current_words, current_tags, current_embeds)

if __name__=="__main__":
    allsents=[]
    unique_tokens=set()
    unique_tokens_lower=set()
    for words, tags in read_conll_file(sys.argv[1]):
        allsents.append(words)
        unique_tokens.update(words)
        unique_tokens_lower.update([w.lower() for w in words])
    print(allsents[1])
    print(allsents[-1])
    assert(len(allsents)==412)
#    assert(len(unique_tokens)==17552)

