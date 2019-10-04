#TODO support batches, 
#TODO perhaps use only BERT?, bert-as-a-service seems a bit overkill
import sys
import os
from bert_serving.server.helper import get_args_parser
from bert_serving.server import BertServer
from bert_serving.client import BertClient
from bert_serving.server.bert import tokenization
import time

if len(sys.argv) < 4:
    print('please provide embeddings, conl file and port')
    exit(0)

port1 = int(sys.argv[3])
port2 = port1 + 1

args = get_args_parser().parse_args(['-model_dir', sys.argv[1],
                                     '-port', str(port1),
                                     '-port_out', str(port2),
                                     '-max_seq_len', 'NONE',
                                     '-pooling_strategy', 'NONE',
                                     '-mask_cls_sep',
                                     '-cpu'])
print('starting bert')
server = BertServer(args)
server.start()
print('started')

#os.system('bert-serving-start -pooling_strategy NONE -model_dir ' + sys.argv[1] + '  -num_worker=1 > /dev/null 2> /dev/null &')

time.sleep(30)#is this necessary?

print('starting client')
bc = BertClient(port=port1, port_out=port2)
print('done')
time.sleep(30)# is this necessary?

curSent = []
outFile = open(sys.argv[2] + '.bert', 'w')
tokenizer = tokenization.FullTokenizer(vocab_file='embeds/bert/vocab.txt')
for line in open(sys.argv[2]):
    if len(line) < 2:
        orig_tokens = [x[0] for x in curSent]
        bert_tokens = []
        # Token map will be an int -> int mapping between the `orig_tokens` index and
        # the `bert_tokens` index.
        orig_to_tok_map = []
        bert_tokens.append("[CLS]")
        for orig_token in orig_tokens:
            orig_to_tok_map.append(len(bert_tokens))
            bert_tokens.extend(tokenizer.tokenize(orig_token))
        bert_tokens.append("[SEP]")
        embs = bc.encode([bert_tokens], is_tokenized=True)[0]
        for wordIdx in range(len(curSent)):
            embStr = 'emb=' + ','.join([str(x) for x in embs[orig_to_tok_map[wordIdx]]])
            outFile.write('\t'.join(curSent[wordIdx] + [embStr]) + '\n')
        outFile.write('\n')
        curSent = []
    else:
        tok = line.strip().split('\t')
        curSent.append(tok)
outFile.close()
os.system('bert-serving-terminate -port ' + str(port1))

