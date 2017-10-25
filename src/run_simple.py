#### Example of using bilty from within code
## 
## to properly seed dyNet add parameter to your script:
## python run_simply.py --dynet-seed 113

from simplebilty import SimpleBiltyTagger
from simplebilty import load_tagger, save_tagger

import random
### Use --dynet-seed $SEED
seed=113 # assume we pass this to script
train_data = "data/da-ud-dev.conllu"
dev_data = "data/da-ud-test.conllu"
in_dim=64
h_dim=100
c_in_dim=100
h_layers=1
iters=2
trainer="sgd"
tagger = SimpleBiltyTagger(in_dim, h_dim,c_in_dim,h_layers,embeds_file=None)
train_X, train_Y = tagger.get_train_data(train_data)
tagger.initialize_graph()
tagger.fit(train_X, train_Y, iters, trainer,seed=seed)
test_X, test_Y = tagger.get_data_as_indices(dev_data)
correct, total = tagger.evaluate(test_X, test_Y)
print(correct, total, correct/total)

# test loading/saving
save_tagger(tagger, "tmp")

tagger2= load_tagger("tmp.model")
correct, total = tagger2.evaluate(test_X, test_Y)
print(correct, total, correct/total)
