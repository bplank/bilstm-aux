## bi-LSTM tagger

Bidirectional Long-Short Term Memory tagger (http://arxiv.org/abs/1604.05529).

### Requirements

* python3 
* [pycnn](https://github.com/clab/cnn)

## Installation

Download and install cnn in a directory of your choice CNNDIR: 

```
mkdir $CNNDIR
git clone https://github.com/clab/cnn
```

Follow the instructions in the Installation readme. However, after
compiling cnn and before compiling pycnn, apply the following patch
(as bilty uses python3): 

``` 
cp pycnn_py3_patch.diff $CNNDIR
cd $CNNDIR
git apply pycnn_py3_patch.diff
```

And compile pycnn:

`make`

After successful installation open python and import pycnn, you should
see something like:

```
>>> import pycnn
[cnn] random seed: 2809331847
[cnn] allocating memory: 512MB
[cnn] memory allocation done.

```


#### Todo:

* make predictions available
* make models available