import sys
import pickle

if len(sys.argv) < 3:
    print('please provide embeddings and pos conl file')
    exit(0)

def loadEmbs(path):
    print('Loading ' + path + '...')
    words, vect = pickle.load(open(path, 'rb'), encoding='latin1')
    embs = {}
    for i in range(len(words)):
        embs[words[i]] = vect[i]
    return embs
embs = loadEmbs(sys.argv[1])

unk = '<UNK>'

outFile = open(sys.argv[2] + '.poly', 'w')
curSent = ''
for line in open(sys.argv[2]):
    if len(line) < 2:
        outFile.write(curSent + '\n')
        curSent = '' 
    else:
        tok = line.strip().split('\t')
        if tok[0] in embs:
            emb = embs[tok[0]]
        elif tok[0].lower() in embs:
            emb = embs[tok[0].lower()]
        else:
            emb = embs[unk]

        embStr = 'emb=' + ','.join([str(x) for x in emb])
        curSent += '\t'.join(tok + [embStr]) + '\n'

outFile.close()
