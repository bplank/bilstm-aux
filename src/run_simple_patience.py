#### Example of using bilty from within code
## 
## to properly seed dyNet add parameter to your script:
## python run_simply.py --dynet-seed 113

from simplebilty import SimpleBiltyTagger
from simplebilty import load_tagger, save_tagger

### Use --dynet-seed $SEED
train_data = "data/da-ud-train.conllu"
test_data = "data/da-ud-test.conllu"
dev_data = "data/da-ud-dev.conllu"
in_dim=64
h_dim=100
c_in_dim=100
h_layers=1
iters=50
trainer="sgd"
tagger = SimpleBiltyTagger(in_dim, h_dim,c_in_dim,h_layers,embeds_file=None)
train_X, train_Y = tagger.get_train_data(train_data)
dev_X, dev_Y = tagger.get_data_as_indices(dev_data)
tagger.initialize_graph()
tagger.fit(train_X, train_Y, iters, val_X=dev_X, val_Y=dev_Y, patience=2, model_path="tmp")
# make sure to reload to get patience 2 model
tagger = load_tagger("tmp")
test_X, test_Y = tagger.get_data_as_indices(test_data)
correct, total = tagger.evaluate(test_X, test_Y)
print(correct, total, correct/total)
