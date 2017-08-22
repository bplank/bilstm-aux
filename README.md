## bi-LSTM tagger

Bidirectional Long-Short Term Memory tagger 

If you use this tagger please cite our paper:
http://arxiv.org/abs/1604.05529

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

#### DyNet supports python 3

The old bilstm-aux had a patch to work with python 3. This
is no longer necessary, as DyNet supports python 3 as of
https://github.com/clab/dynet/pull/130#issuecomment-259656695


#### Example command

Training the tagger:

```
python src/bilty.py --dynet-mem 1500 --train data/da-ud-train.conllu --dev data/da-ud-test.conllu --iters 10 --pred_layer 1
```

#### Embeddings

The Polyglot embeddings [(Al-Rfou et al.,
2013)](https://sites.google.com/site/rmyeid/projects/polyglot) can be
downloaded from [here](http://www.let.rug.nl/bplank/bilty/embeds.tar.gz) (0.6GB)

### Options:

You can see the options by running:

```
python src/bilty.sh --help
```

A great option is DyNet autobatching [Neubig et al., 2017](https://arxiv.org/abs/1705.07860). It speeds up training by ~20\%. 
You can activate it with:

``
--dynet-autobatch
``

#### A couple of remarks

The choice of 22 languages from UD1.2 (rather than 33) is described in
our TACL parsing paper, Section 3.1. [(Agić et al.,
2016)](https://transacl.org/ojs/index.php/tacl/article/view/869). Note,
however, that the bi-LSTM tagger does not require large amounts of
training data (as discussed in our ACL 2016 paper). 

The `bilty` code is a significantly refactored version of the code
originally used in the paper. For example, `bilty` supports multi-task
learning with output layers at different layers (`--pred_layer`), as
well as stacked LSTMs (see e.g., Ballesteros et al., 2015, Dyer et
al., 2015). 

DyNet 2.0 switched the default LSTM implementation to
VanillaLSTMBuilder.  We observe slightly higher POS tagging results
with the earlier CoupledLSTMBuilder, which is the current default.

#### Reference

```
@inproceedings{plank:ea:2016,
  title={{Multilingual Part-of-Speech Tagging with Bidirectional Long Short-Term Memory Models and Auxiliary Loss}},
  author={Plank, Barbara and S{\o}gaard, Anders and Goldberg, Yoav},
  booktitle={ACL 2016, arXiv preprint arXiv:1604.05529},
  url={http://arxiv.org/abs/1604.05529},
  year={2016}
}
```

