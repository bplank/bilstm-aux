#TODO support batches

import sys
import os
sys.path.append('.')
from ELMoForManyLangs.elmoformanylangs import Embedder

if len(sys.argv) < 3:
    print('please provide embeddings and conl file')
    exit(0)

converter = Embedder(sys.argv[1])
curSent = []
outFile = open(sys.argv[2] + '.elmo', 'w')
for line in open(sys.argv[2]):
    if len(line) < 2:
        sent = [[x[0] for x in curSent]]
        emb = converter.sents2elmo(sent)[0]
        for itemIdx in range(len(curSent)):
            embStr = 'emb=' + ','.join([str(x) for x in emb[itemIdx]])
            outFile.write('\t'.join(curSent[itemIdx] + [embStr]) + '\n')
        outFile.write('\n')
        curSent = []
    else:
        tok = line.strip().split('\t')
        curSent.append(tok)

outFile.close()
