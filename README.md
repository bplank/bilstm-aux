## bi-LSTM sequence tagger 

Bidirectional Long-Short Term Memory sequence tagger 

This is a new version (`structbilty`) of an earlier bi-LSTM tagger (Plank et al., 2016) based on our EMNLP 2018 paper (DsDs).

If you use this tagger please cite our papers:

* https://aclanthology.coli.uni-saarland.de/papers/D18-1061/d18-1061
* http://arxiv.org/abs/1604.05529

### Requirements

* python3 
* [DyNet 2.0](https://github.com/clab/dynet)

## Installation

Download and install dynet in a directory of your choice DYNETDIR: 

```
mkdir $DYNETDIR
git clone https://github.com/clab/dynet
```

Follow the instructions in the Dynet documentation (use `-DPYTHON`,
see http://dynet.readthedocs.io/en/latest/python.html). 

And compile dynet:

```
cmake .. -DEIGEN3_INCLUDE_DIR=$HOME/tools/eigen/ -DPYTHON=`which python`
```

(if you have a GPU, use: [note: non-deterministic behavior]):

```
cmake .. -DEIGEN3_INCLUDE_DIR=$HOME/tools/eigen/ -DPYTHON=`which python` -DBACKEND=cuda
```


After successful installation open python and import dynet, you can
test if the installation worked with:

```
>>> import dynet
[dynet] random seed: 2809331847
[dynet] allocating memory: 512MB
[dynet] memory allocation done.
>>> dynet.__version__
2.0
```

(You may need to set you PYTHONPATH to include Dynet's `build/python`)


#### Example command

Training the tagger:

```
python src/structbilty.py --dynet-mem 1500 --train data/da-ud-train.conllu --test data/da-ud-test.conllu --iters 10 --model da
```

Training with patience:
```
python src/structbilty.py --dynet-mem 1500 --train data/da-ud-train.conllu --dev data/da-ud-dev.conllu --test data/da-ud-test.conllu --iters 50 --model da --patience 2
```

Testing:
```
python src/structbilty.py --model da --test data/da-ud-test.conllu --output predictions/test-da.out
```

Training and testing in two steps (`--model` for both saving and loading):

```
mkdir -p predictions
python src/structbilty.py --dynet-mem 1500 --train data/da-ud-train.conllu --iters 10 --model da

python src/structbilty.py --model da --test data/da-ud-test.conllu --output predictions/test-da.out
```

#### Embeddings

The Polyglot embeddings [(Al-Rfou et al.,
2013)](https://sites.google.com/site/rmyeid/projects/polyglot) can be
downloaded from [here](http://www.let.rug.nl/bplank/bilty/embeds.tar.gz) (0.6GB)

### Options:

You can see the options by running:

```
python src/structbilty.py --help
```

A great option is DyNet autobatching ([Neubig et al.,
2017](https://arxiv.org/abs/1705.07860)).  It speeds up training considerably (
~20\%).  You can activate it with:

``
python src/structbilty.sh --dynet-autobatch 1
``

#### Major changes:

- major refactoring of internal data handling
- renaming to `structbilty`
- `--pred-layer` is no longer required
- a single `--model` options handles both saving and loading model parameters
- the option of running a CRF has been added
- the tagger can handle additional lexical features (see our DsDs paper, EMNLP 2018) below 
- grouping of arguments
- `simplebilty` is deprecated (still available in the [former release](https://github.com/bplank/bilstm-aux/releases/tag/v1.0)

#### Todo

- move to DyNet 2.1

#### References

```
@InProceedings{plank-agic:2018,
  author = 	"Plank, Barbara
		and Agi{\'{c}}, {\v{Z}}eljko",
  title = 	"Distant Supervision from Disparate Sources for Low-Resource Part-of-Speech Tagging",
  booktitle = 	"Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing",
  year = 	"2018",
  publisher = 	"Association for Computational Linguistics",
  pages = 	"614--620",
  location = 	"Brussels, Belgium",
  url = 	"http://aclweb.org/anthology/D18-1061"
}

@inproceedings{plank:ea:2016,
  title={{Multilingual Part-of-Speech Tagging with Bidirectional Long Short-Term Memory Models and Auxiliary Loss}},
  author={Plank, Barbara and S{\o}gaard, Anders and Goldberg, Yoav},
  booktitle={ACL 2016, arXiv preprint arXiv:1604.05529},
  url={http://arxiv.org/abs/1604.05529},
  year={2016}
}
```

