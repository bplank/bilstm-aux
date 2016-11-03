## bi-LSTM tagger

Bidirectional Long-Short Term Memory tagger 

If you use this tagger please cite our paper:
http://arxiv.org/abs/1604.05529

### Requirements

* python3 
* [dynet](https://github.com/clab/dynet)

## Installation

Download and install dynet in a directory of your choice DYNETDIR: 

```
mkdir $DYNETDIR
git clone https://github.com/clab/dynet
```

Follow the instructions in the Dynet documentation (use `-DPYTHON`,
see http://dynet.readthedocs.io/en/latest/python.html). However, after
compiling DyNet and before compiling the Python binding, apply the
following patch (as bilty uses python3):

``` 
cp dynet_py3_patch.diff $DYNETDIR
cd $DYNETDIR
git apply dynet_py3_patch.diff
```

And compile dynet:

```
cmake .. -DEIGEN3_INCLUDE_DIR=$HOME/tools/eigen/ -DPYTHON=`which python`
```

(if you have a GPU:

```
cmake .. -DEIGEN3_INCLUDE_DIR=$HOME/tools/eigen/ -DPYTHON=`which python` -DBACKEND=cuda
```
)

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

#### Results on UD1.3

NB. The results below are with the previous version of Dynet (pycnn).

The table below provides results on UD1.3 (iters=20, h_layers=1).

+poly is using pre-trained embeddings to initialize
word embeddings.  Note that for some languages it slightly hurts performance.

```
python src/bilty.py --dynet-seed 1512141834 --dynet-mem 1500 --train /home/$user/corpora/pos/ud1.3/orgtok/goldpos//en-ud-train.conllu --test /home/$user/corpora/pos/ud1.3/orgtok/goldpos//en-ud-test.conllu --dev /home/$user/corpora/pos/ud1.3/orgtok/goldpos//en-ud-dev.conllu --output /data/$user/experiments/bilty/predictions/bilty/en-ud-test.conllu.bilty-en-ud1.3-poly-i20-h1 --in_dim 64 --c_in_dim 100 --trainer sgd --iters 20 --sigma 0.2 --save /data/$user/experiments/bilty/models/bilty/bilty-en-ud1.3-poly-i20-h1.model --embeds embeds/poly_a/en.polyglot.txt --h_layers 1 --pred_layer 1  > /data/$user/experiments/bilty/nohup/bilty-en-ud1.3-poly-i20-h1.out 2> /data/$user/experiments/bilty/nohup/bilty.bilty-en-ud1.3-poly-i20-h1.out2
```

| Lang | i20-h1  | +poly |
| ---| -----:| -----:|
| ar | 96.07 | 96.37 |
| bg | 98.21 | 98.12 |
| ca | 98.11 | 98.24 |
| cs | 98.63 | 98.60 |
| cu | 96.48 | -- |
| da | 96.06 | 96.04 |
| de | 92.91 | 93.64 |
| el | 97.85 | 98.36 |
| en | 94.60 | 95.04 |
| es | 95.23 | 95.76 |
| et | 95.75 | 96.57 |
| eu | 93.86 | 95.40 |
| fa | 96.82 | 97.38 |
| fi | 94.32 | 95.35 |
| fr | 96.34 | 96.45 |
| ga | 90.50 | 91.29 |
| gl | 96.89 | -- |
| got | 95.97 | -- |
| grc | 94.36 | -- |
| he | 95.25 | 96.78 |
| hi | 96.37 | 96.93 |
| hr | 94.98 | 96.07 |
| hu | 93.84 | -- |
| id | 93.17 | 93.55 |
| it | 97.40 | 97.82 |
| kk | 77.68 | -- |
| la | 90.17 | -- |
| lv | 91.42 | -- |
| nl | 90.02 | 89.87 |
| no | 97.58 | 97.97 |
| pl | 96.30 | 97.36 |
| pt | 97.21 | 97.46 |
| ro | 95.49 | -- |
| ru | 95.69 | -- |
| sl | 97.53 | 96.42 |
| sv | 96.49 | 96.76 |
| ta | 84.51 | -- |
| tr | 93.81 | -- |
| zh | 93.13 | -- |

Using pre-trained embeddings often helps to improve accuracy, however, does not
strictly hold for all languages.

For more information, predictions files and pre-trained models
visit [http://www.let.rug.nl/bplank/bilty/](http://www.let.rug.nl/bplank/bilty/)

#### Embeddings

The poly embeddings [(Al-Rfou et al.,
2013)](https://sites.google.com/site/rmyeid/projects/polyglot) can be
downloaded from [here](http://www.let.rug.nl/bplank/bilty/embeds.tar.gz) (0.6GB)


#### A couple of remarks

The choice of 22 languages from UD1.2 (rather than 33) is described in
our TACL parsing paper, Section 3.1. [(Agić et al.,
2016)](https://transacl.org/ojs/index.php/tacl/article/view/869). Note,
however, that the bi-LSTM tagger does not require large amounts of
training data (as discussed in our paper). Therefore above are 
results for all languages in UD1.3 (for the canonical language
subparts, i.e., those with just the language prefix, no further
suffix; e.g. 'nl' but not 'nl_lassy', and those languages which are
distributed with word forms).

The `bilty` code is a significantly refactored version of the code
originally used in the paper. For example, `bilty` supports multi-task
learning with output layers at different layers (`--pred_layer`), and
it correctly supports stacked LSTMs (see e.g., Ballesteros et al.,
2015, Dyer et al., 2015). The results on UD1.3 are obtained with
`bilty` using no stacking (`--h_layers 1`). 

#### Recommended setting for `bilty`:

* 3 stacked LSTMs, predicting on outermost layer, otherwise default settings, i.e., `--h_layers 3 --pred_layer 3`

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

