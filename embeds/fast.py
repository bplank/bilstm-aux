import sys
import os
if len(sys.argv) < 3:
    print('please provide embeddings and pos conl file')
    exit(0)


#TODO could be more efficient by only generating for unique words? (probably fastText already does that?)

posFile = sys.argv[2]
embFile = posFile + '.embeds'
wordFile = posFile + '.words'

cmd = 'cut -f 1 ' + posFile + ' | grep -v "^$" > ' + wordFile
print(cmd)
os.system(cmd)

cmd = "cd fastText && ./fasttext print-word-vectors ../" + sys.argv[1] + ' < ../' + wordFile + ' > ../' + embFile + ' && cd ../'
print(cmd)
os.system(cmd)

embs = []
for line in open(embFile):
    line = line.strip()[line.find(' ')+1:].replace(' ',',')
    embs.append(line)

outFile = open(posFile + '.fast', 'w')
wordIdx = 0
for line in open(posFile):
    if len(line) <= 2:
        outFile.write(line)
    else:
        tok = line.strip().split('\t')
        embStr = 'emb=' + embs[wordIdx]
        outFile.write('\t'.join(tok + [embStr]) + '\n')
        wordIdx += 1
outFile.close()

cmd = 'rm ' + embFile + ' ' + wordFile
print(cmd)
os.system(cmd)

