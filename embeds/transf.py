# This script is built for transformers 4.0.0, it needs adaptations for older versions

# based on https://github.com/huggingface/transformers/blob/master/notebooks/02-transformers.ipynb
# TODO could probably be faster when batching is used?
# TODO apparently running on gpu is trivial? (https://github.com/huggingface/transformers/issues/2704)

import sys
from transformers import AutoModel, AutoTokenizer

if len(sys.argv) < 3:
    print(
        "please provide embeddings name (from https://huggingface.co/models) and pos conll file"
    )
    exit(0)
elif len(sys.argv) == 3:
    # python transf.py CONLL_FILE TRANSFORMER_NAME
    model = AutoModel.from_pretrained(sys.argv[2])
    tokenizer = AutoTokenizer.from_pretrained(sys.argv[2])
elif len(sys.argv) == 4:
    # python transf.py CONLL_FILE TRANSFORMER_INDEX TRANSFORMER_VOCAB
    model = AutoModel.from_pretrained(sys.argv[2])
    tokenizer = AutoTokenizer.from_pretrained(sys.argv[3])
elif len(sys.argv) == 5:
    # python transf.py CONLL_FILE TRANSFORMER_INDEX TRANSFORMER_VOCAB TRANSFORMER_CONFIG
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(sys.argv[4])
    model = AutoModel.from_pretrained(sys.argv[2], config=config)
    tokenizer = AutoTokenizer.from_pretrained(sys.argv[3])
else:
    print("Too many arguments, can't make sense of this.")
    exit(0)


def sentToEmbed(sent):
    tokens_pt = tokenizer(
        sent, return_tensors="pt", return_offsets_mapping=True, is_split_into_words=True
    )

    # first dimension=first sent. Second dimensions = remove special tokens
    offsets = tokens_pt["offset_mapping"][0][1:-1]

    # first dimension, get full output (not pool) second dimension, get first sentence, third dimension remove special tokens
    outputs = model(tokens_pt["input_ids"])[0][0][1:-1]
    # size of outputs is now the number of wordpieces without special tokens, 2nd dimensions is embeddings size

    # get all indexes that should be kept
    startOfWords = []
    for offsetIdx, offset in enumerate(offsets):
        if offset[0] == 0:
            startOfWords.append(offsetIdx)

    # keep only the first embedding of each word
    return outputs[startOfWords, :]


outFile = open(sys.argv[1] + "." + sys.argv[2], "w")
curSent = []
for line in open(sys.argv[1]):
    line = line.strip("\n")
    if len(line) < 2:
        embeds = sentToEmbed([word[0] for word in curSent])
        for word, embed in zip(curSent, embeds):
            embStr = "emb=" + ",".join([str(float(x)) for x in embed])
            outFile.write("\t".join(word + [embStr]) + "\n")
        outFile.write("\n")
        curSent = []
    else:
        tok = line.strip().split("\t")
        curSent.append(tok)

outFile.close()
