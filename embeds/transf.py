#based on https://github.com/huggingface/transformers/blob/master/notebooks/02-transformers.ipynb
import torch
from transformers import AutoModel, AutoTokenizer


# Store the model we want to use
MODEL_NAME = "bert-base-cased"

model = AutoModel.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def sentToEmbed(sent):
    tokens_pt = tokenizer(sent, return_tensors="pt", return_offsets_mapping=True, is_split_into_words=True)

    # first dimension=first sent. Second dimensions = remove special tokens
    offsets = tokens_pt['offset_mapping'][0][1:-1] 
    
    # first dimension, get full output (not pool) second dimension, get first sentence, third dimension remove special tokens
    outputs = model(tokens_pt['input_ids'])[0][0][1:-1]
    # size of outputs is now the number of wordpieces without special tokens, 2nd dimensions is embeddings size
    
    # get all indexes that should be kept
    startOfWords = []
    for offsetIdx, offset in enumerate(offsets):
        if offset[0] == 0:
            startOfWords.append(offsetIdx)
    
    # keep only the first embedding of each word
    return outputs[startOfWords, :]

sent = ['ThisThisThis', 'is', 'an', 'input', 'exampleexample']

embed = sentToEmbed(sent)
print(embed.shape)
    
