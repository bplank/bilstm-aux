import sys
from embeddings import GloveEmbedding

if len(sys.argv) < 3:
    print("please provide embeddings and pos conl file")
    exit(0)


embs = GloveEmbedding(sys.argv[1], default="random")

unk = "<UNK>"

outFile = open(sys.argv[2] + ".glove", "w")
curSent = ""
for line in open(sys.argv[2]):
    if len(line) < 2:
        outFile.write(curSent + "\n")
        curSent = ""
    else:
        tok = line.strip().split("\t")
        emb = embs.emb(tok[0])

        embStr = "emb=" + ",".join([str(x) for x in emb])
        curSent += "\t".join(tok + [embStr]) + "\n"

outFile.close()