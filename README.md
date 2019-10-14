## bi-LSTM sequence tagger 

Bidirectional Long-Short Term Memory sequence tagger 

This is a new extended version (`structbilty`) of the earlier bi-LSTM tagger by Plank et al., (2016).

If you use this tagger please (cite)[http://arxiv.org/abs/1604.05529]:

```
@inproceedings{plank-etal-2016,
    title = "Multilingual Part-of-Speech Tagging with Bidirectional Long Short-Term Memory Models and Auxiliary Loss",
    author = "Plank, Barbara  and
      S{\o}gaard, Anders  and
      Goldberg, Yoav",
    booktitle = "Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)",
    month = aug,
    year = "2016",
    address = "Berlin, Germany",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/P16-2067",
    doi = "10.18653/v1/P16-2067",
    pages = "412--418",
}
```


For the version called DsDs, please cite: https://aclanthology.coli.uni-saarland.de/papers/D18-1061/d18-1061


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
downloaded from [here](http://www.itu.dk/people/bapl/embeds.tar.gz) (0.6GB)

You can load generic word embeddings by using --embeds, however Bilty also supports loading embeddings from the input files. This can be enabled by --embeds_in_file and expects the train/dev/test files to be in the following format:

```
word1<tab>tag1<tab>emb=val1,val2,val3,...
word2<tab>tag1<tab>emb=val1,val2,val3,...
...
```

Note that the dimensions of embeddings should match the --in_dim.

We also provide scripts to generate these files for four commonly used embeddings types (Polyglot, Fasttext, ELMo and BERT), which can be found in the `embeds` folder. If we for example want to use polyglot embeddings we need to run the following commands:

```
python3 embeds/poly.prep.py 
python3 embeds/poly.py embeds/polyglot/nl.pickle nl.train.pos 
python3 embeds/poly.py embeds/polyglot/nl.pickle nl.dev.pos 
python3 embeds/poly.py embeds/polyglot/nl.pickle nl.test.pos 
``` 

This creates .poly files which can be used as input to Bilty when --words_in_file is enabled. For now the language is hardcoded in these scripts, please modify `*.prep.py` accordingly.


please note that this option does not support updating embeddings (so you have to use --disable-backprob-embeds), and also does not support the --raw option.

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
- `simplebilty` is deprecated (still available in the [former release](https://github.com/bplank/bilstm-aux/releases/tag/v1.0))

#### Todo

- move to DyNet 2.1

#### References

```
@inproceedings{plank-etal-2016,
    title = "Multilingual Part-of-Speech Tagging with Bidirectional Long Short-Term Memory Models and Auxiliary Loss",
    author = "Plank, Barbara  and
      S{\o}gaard, Anders  and
      Goldberg, Yoav",
    booktitle = "Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)",
    month = aug,
    year = "2016",
    address = "Berlin, Germany",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/P16-2067",
    doi = "10.18653/v1/P16-2067",
    pages = "412--418",
}

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


```

