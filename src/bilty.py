#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A neural network based tagger (bi-LSTM)
- hierarchical (word embeddings plus lower-level bi-LSTM for characters)
- supports MTL
:author: Barbara Plank
"""
import argparse
import random
import time
import sys
import numpy as np
import os
import pickle
import dynet
import codecs

from sklearn.model_selection import train_test_split

from collections import Counter, defaultdict
from lib.mnnl import FFSequencePredictor, Layer, RNNSequencePredictor, BiRNNSequencePredictor
from lib.mio import read_conll_file, load_embeddings_file

UNK = "_UNK"

from lib.mmappers import TRAINER_MAP, ACTIVATION_MAP, INITIALIZER_MAP, BUILDERS

def main():
    parser = argparse.ArgumentParser(description="""Run the NN tagger""")
    parser.add_argument("--train", nargs='*', help="train folder for each task") # allow multiple train files, each asociated with a task = position in the list
    parser.add_argument("--pred_layer", nargs='*', help="layer of predictons for each task", required=True) # for each task the layer on which it is predicted (default 1)
    parser.add_argument("--model", help="load model from file", required=False)
    parser.add_argument("--iters", help="training iterations [default: 30]", required=False,type=int,default=30)
    parser.add_argument("--in_dim", help="input dimension [default: 64] (like Polyglot embeds)", required=False,type=int,default=64)
    parser.add_argument("--c_in_dim", help="input dimension for character embeddings [default: 100]", required=False,type=int,default=100)
    parser.add_argument("--h_dim", help="hidden dimension [default: 100]", required=False,type=int,default=100)
    parser.add_argument("--h_layers", help="number of stacked LSTMs [default: 1 = no stacking]", required=False,type=int,default=1)
    parser.add_argument("--test", nargs='*', help="test file(s)", required=False) # should be in the same order/task as train
    parser.add_argument("--raw", help="if test file is in raw format (one sentence per line)", required=False, action="store_true", default=False)
    parser.add_argument("--dev", help="dev file(s)", required=False) 
    parser.add_argument("--output", help="output predictions to file", required=False,default=None)
    parser.add_argument("--output-probs", help="output prediction probs to file (last column)", required=False, default=None)
    parser.add_argument("--save", help="save model to file (appends .model as well as .pickle)",default=None)
    parser.add_argument("--embeds", help="word embeddings file", required=False, default=None)
    parser.add_argument("--sigma", help="noise sigma", required=False, default=0.2, type=float)
    parser.add_argument("--ac", help="activation function [rectify, tanh, ...]", default="tanh", choices=ACTIVATION_MAP.keys())
    parser.add_argument("--mlp", help="use MLP layer of this dimension [default 0=disabled]", required=False, default=0, type=int)
    parser.add_argument("--ac-mlp", help="activation function for MLP (if used) [rectify, tanh, ...]", default="rectify", choices=ACTIVATION_MAP.keys())
    parser.add_argument("--trainer", help="trainer [default: sgd]", required=False, choices=TRAINER_MAP.keys(), default="sgd")
    parser.add_argument("--learning-rate", help="learning rate [0: use default]", default=0, type=float) # see: http://dynet.readthedocs.io/en/latest/optimizers.html
    parser.add_argument("--patience", help="patience [default: 0=not used], requires specification of --dev and model path --save", required=False, default=0, type=int)
    parser.add_argument("--log-losses", help="log loss (for each task if multiple active)", required=False, action="store_true", default=False)
    parser.add_argument("--word-dropout-rate", help="word dropout rate [default: 0.25], if 0=disabled, recommended: 0.25 (Kipperwasser & Goldberg, 2016)", required=False, default=0.25, type=float)

    parser.add_argument("--dynet-seed", help="random seed for dynet (needs to be first argument!)", required=False, type=int)
    parser.add_argument("--dynet-mem", help="memory for dynet (needs to be first argument!)", required=False, type=int)
    parser.add_argument("--dynet-gpus", help="1 for GPU usage", default=0, type=int) # warning: non-deterministic results on GPU https://github.com/clab/dynet/issues/399
    parser.add_argument("--dynet-autobatch", help="if 1 enable autobatching", default=0, type=int)
    parser.add_argument("--minibatch-size", help="size of minibatch for autobatching (1=disabled)", default=1, type=int)

    parser.add_argument("--save-embeds", help="save word embeddings file", required=False, default=None)
    parser.add_argument("--disable-backprob-embeds", help="disable backprob into embeddings (default is to update)", required=False, action="store_false", default=True)
    parser.add_argument("--initializer", help="initializer for embeddings (default: constant)", choices=INITIALIZER_MAP.keys(), default="constant")
    parser.add_argument("--builder", help="RNN builder (default: lstmc)", choices=BUILDERS.keys(), default="lstmc")

    # new parameters
    parser.add_argument('--max-vocab-size', type=int, help='the maximum size '
                                                           'of the vocabulary')

    args = parser.parse_args()

    if args.output is not None:
        if os.path.isdir(args.output):
            if not os.path.exists(args.output):
                os.mkdir(os.path.dirname(args.output))

    if args.save:
        model_dirname = os.path.dirname(args.save)
        if os.path.isdir(model_dirname) and not os.path.exists(model_dirname):
            os.mkdir(model_dirname)

    if args.dev:
        assert (args.save is not None), "Require to save model to --save MODELNAME when --dev is given"

    if args.train:
        if not args.pred_layer:
            print("--pred_layer required!")
            exit()
    
    if args.dynet_seed:
        print(">>> using seed: {} <<< ".format(args.dynet_seed), file=sys.stderr)
        np.random.seed(args.dynet_seed)
        random.seed(args.dynet_seed)

    if args.c_in_dim == 0:
        print(">>> disable character embeddings <<<", file=sys.stderr)

    if args.minibatch_size > 1:
        print(">>> using minibatch_size {} <<<".format(args.minibatch_size))

    if args.patience:
        if not args.dev or not args.save:
            print("patience requires a dev set and model path (--dev and --save)")
            exit()

    if args.save:
        # check if folder exists
        if os.path.isdir(args.save):
            if not os.path.exists(args.save):
                print("Creating {}..".format(args.save))
                os.makedirs(args.save)

    if args.output:
        if os.path.isdir(args.output):
            outdir = os.path.dirname(args.output)
            if not os.path.exists(outdir):
                os.makedirs(outdir)

    start = time.time()

    if args.model:
        print("loading model from file {}".format(args.model), file=sys.stderr)
        tagger = load(args.model)
    else:
        tagger = NNTagger(args.in_dim,
                          args.h_dim,
                          args.c_in_dim,
                          args.h_layers,
                          args.pred_layer,
                          embeds_file=args.embeds,
                          activation=ACTIVATION_MAP[args.ac],
                          mlp=args.mlp,
                          activation_mlp=ACTIVATION_MAP[args.ac_mlp],
                          noise_sigma=args.sigma,
                          learning_algo=args.trainer,
                          learning_rate=args.learning_rate,
                          backprob_embeds=args.disable_backprob_embeds,
                          initializer=INITIALIZER_MAP[args.initializer],
                          builder=BUILDERS[args.builder],
                          max_vocab_size=args.max_vocab_size
                          )

    if args.train and len( args.train ) != 0:
        tagger.fit(args.train, args.iters,
                   dev=args.dev, word_dropout_rate=args.word_dropout_rate,
                   model_path=args.save, patience=args.patience, minibatch_size=args.minibatch_size, log_losses=args.log_losses)

        if args.save and not args.patience:  # in case patience is active it gets saved in the fit function
            save(tagger, args.save)
            tagger = load(args.save)

        if args.patience:
            # reload patience 2 model
            tagger = load(args.save)


    if args.test and len( args.test ) != 0:
        if not args.model:
            if not args.train:
                print("specify a model!")
                sys.exit()

        stdout = sys.stdout
        # One file per test ... 
        for i, test in enumerate(args.test):

            if args.output is not None:
                file_pred = args.output+".task"+str(i)
                sys.stdout = codecs.open(file_pred, 'w', encoding='utf-8')

            sys.stderr.write('\nTesting Task'+str(i)+'\n')
            sys.stderr.write('*******\n')
            test_X, test_Y, org_X, org_Y, task_labels = tagger.get_data_as_indices(test, "task"+str(i), raw=args.raw)
            correct, total = tagger.evaluate(test_X, test_Y, org_X, org_Y, task_labels,
                                             output_predictions=args.output, output_probs=args.output_probs, raw=args.raw)

            if not args.raw:
                print("\nTask%s test accuracy on %s items: %.4f" % (i, i+1, correct/total), file=sys.stderr)
            print(("Done. Took {0:.2f} seconds.".format(time.time()-start)),file=sys.stderr)
            sys.stdout = stdout
    if args.train:
        print("Info: biLSTM\n\t"+"\n\t".join(["{}: {}".format(a,v) for a, v in vars(args).items()
                                          if a not in ["train","test","dev","pred_layer"]]))
    else:
        # print less when only testing, as not all train params are stored explicitly
        print("Info: biLSTM\n\t" + "\n\t".join(["{}: {}".format(a, v) for a, v in vars(args).items()
                                                if a not in ["train", "test", "dev", "pred_layer",
                                                             "initializer","ac","word_dropout_rate",
                                                             "patience","sigma","disable_backprob_embed",
                                                             "trainer", "dynet_seed", "dynet_mem","iters"]]))

    if args.save_embeds:
        tagger.save_embeds(args.save_embeds)


def load(model_path):
    """
    load a model from file; specify the .model file, it assumes the *pickle file in the same location
    """
    myparams = pickle.load(open(model_path+".params.pickle", "rb"))
    tagger = NNTagger(myparams["in_dim"],
                      myparams["h_dim"],
                      myparams["c_in_dim"],
                      myparams["h_layers"],
                      myparams["pred_layer"],
                      activation=myparams["activation"],
                      mlp=myparams["mlp"],
                      activation_mlp=myparams["activation_mlp"],
                      tasks_ids=myparams["tasks_ids"],
                      builder=myparams["builder"],
                      )
    tagger.set_indices(myparams["w2i"],myparams["c2i"],myparams["task2tag2idx"])
    tagger.predictors, tagger.char_rnn, tagger.wembeds, tagger.cembeds = \
        tagger.build_computation_graph(myparams["num_words"],
                                       myparams["num_chars"])
    tagger.model.populate(model_path+".model")
    print("model loaded: {}".format(model_path), file=sys.stderr)
    return tagger


def save(nntagger, model_path):
    """
    save a model; dynet only saves the parameters, need to store the rest separately
    """
    modelname = model_path + ".model"
    nntagger.model.save(modelname)
    myparams = {"num_words": len(nntagger.w2i),
                "num_chars": len(nntagger.c2i),
                "tasks_ids": nntagger.tasks_ids,
                "w2i": nntagger.w2i,
                "c2i": nntagger.c2i,
                "task2tag2idx": nntagger.task2tag2idx,
                "activation": nntagger.activation,
                "mlp": nntagger.mlp,
                "activation_mlp": nntagger.activation_mlp,
                "in_dim": nntagger.in_dim,
                "h_dim": nntagger.h_dim,
                "c_in_dim": nntagger.c_in_dim,
                "h_layers": nntagger.h_layers,
                "embeds_file": nntagger.embeds_file,
                "pred_layer": nntagger.pred_layer,
                "builder": nntagger.builder,
                }
    pickle.dump(myparams, open( model_path+".params.pickle", "wb" ) )
    print("model stored: {}".format(modelname), file=sys.stderr)


class NNTagger(object):

    def __init__(self,in_dim,h_dim,c_in_dim,h_layers,pred_layer,learning_algo="sgd", learning_rate=0,
                 embeds_file=None,activation=ACTIVATION_MAP["tanh"],mlp=0,activation_mlp=ACTIVATION_MAP["rectify"],
                 backprob_embeds=True,noise_sigma=0.1, tasks_ids=[],
                 initializer=INITIALIZER_MAP["glorot"], builder=BUILDERS["lstmc"],
                 max_vocab_size=None):
        self.w2i = {}  # word to index mapping
        self.c2i = {}  # char to index mapping
        self.tasks_ids = tasks_ids # list of names for each task
        self.task2tag2idx = {} # need one dictionary per task
        self.pred_layer = [int(layer) for layer in pred_layer] # at which layer to predict each task
        self.model = dynet.ParameterCollection() #init model
        self.in_dim = in_dim
        self.h_dim = h_dim
        self.c_in_dim = c_in_dim
        self.activation = activation
        self.mlp = mlp
        self.activation_mlp = activation_mlp
        self.noise_sigma = noise_sigma
        self.h_layers = h_layers
        self.predictors = {"inner": [], "output_layers_dict": {}, "task_expected_at": {} } # the inner layers and predictors
        self.wembeds = None # lookup: embeddings for words
        self.cembeds = None # lookup: embeddings for characters
        self.embeds_file = embeds_file
        trainer_algo = TRAINER_MAP[learning_algo]
        if learning_rate > 0:
            ### TODO: better handling of additional learning-specific parameters
            self.trainer = trainer_algo(self.model, learning_rate=learning_rate)
        else:
            # using default learning rate
            self.trainer = trainer_algo(self.model)
        self.backprob_embeds = backprob_embeds
        self.initializer = initializer
        self.char_rnn = None # biRNN for character input
        self.builder = builder # default biRNN is an LSTM
        self.max_vocab_size = max_vocab_size


    def pick_neg_log(self, pred, gold):
        return -dynet.log(dynet.pick(pred, gold))

    def set_indices(self, w2i, c2i, task2t2i):
        for task_id in task2t2i:
            self.task2tag2idx[task_id] = task2t2i[task_id]
        self.w2i = w2i
        self.c2i = c2i

    def fit(self, list_folders_name, num_iterations, dev=None, word_dropout_rate=0.0, model_path=None, patience=0, minibatch_size=0, log_losses=False):
        """
        train the tagger
        """
        print("read training data",file=sys.stderr)

        nb_tasks = len( list_folders_name )

        losses = {} # log losses

        train_X, train_Y, task_labels, w2i, c2i, task2t2i = self.get_train_data(list_folders_name)

        ## after calling get_train_data we have self.tasks_ids
        self.task2layer = {task_id: out_layer for task_id, out_layer in zip(self.tasks_ids, self.pred_layer)}
        print("task2layer", self.task2layer, file=sys.stderr)

        # store mappings of words and tags to indices
        self.set_indices(w2i,c2i,task2t2i)

        # if we use word dropout keep track of counts
        if word_dropout_rate > 0.0:
            widCount = Counter()
            for sentence, _ in train_X:
                widCount.update([w for w in sentence])

        if dev:
            if not os.path.exists(dev):
                print('%s does not exist. Using 10 percent of the training '
                      'dataset for validation.' % dev)
                train_X, dev_X, train_Y, dev_Y = train_test_split(
                    train_X, train_Y, test_size=0.1)
                org_X, org_Y = None, None
                dev_task_labels = ['task0'] * len(train_X)
            else:
                dev_X, dev_Y, org_X, org_Y, dev_task_labels = self.get_data_as_indices(dev, "task0")

        # init lookup parameters and define graph
        print("build graph",file=sys.stderr)
        
        num_words = len(self.w2i)
        num_chars = len(self.c2i)
        
        assert(nb_tasks==len(self.pred_layer))
        
        self.predictors, self.char_rnn, self.wembeds, self.cembeds = self.build_computation_graph(num_words, num_chars)

        if self.backprob_embeds == False:
            ## disable backprob into embeds (default: True)
            self.wembeds.set_updated(False)
            print(">>> disable wembeds update <<< (is updated: {})".format(self.wembeds.is_updated()), file=sys.stderr)

        train_data = list(zip(train_X,train_Y, task_labels))

        best_val_acc, epochs_no_improvement = 0.0, 0

        if dev and model_path is not None and patience > 0:
            print('Using early stopping with patience of %d...' % patience)

        batch = []

        for iter in range(num_iterations):

            total_loss=0.0
            total_tagged=0.0
            random.shuffle(train_data)

            loss_accum_loss = defaultdict(float)
            loss_accum_tagged = defaultdict(float)

            for ((word_indices,char_indices),y, task_of_instance) in train_data:

                if word_dropout_rate > 0.0:
                    word_indices = [self.w2i[UNK] if
                                        (random.random() > (widCount.get(w)/(word_dropout_rate+widCount.get(w))))
                                        else w for w in word_indices]

                if task_of_instance not in losses:
                    losses[task_of_instance] = [] #initialize

                if minibatch_size > 1:
                    # accumulate instances for minibatch update
                    output = self.predict(word_indices, char_indices, task_of_instance, train=True)
                    total_tagged += len(word_indices)

                    loss1 = dynet.esum([self.pick_neg_log(pred,gold) for pred, gold in zip(output, y)])
                    batch.append(loss1)
                    if len(batch) == minibatch_size:
                        loss = dynet.esum(batch)
                        total_loss += loss.value()

                        # logging
                        loss_accum_tagged[task_of_instance] += len(word_indices)
                        loss_accum_loss[task_of_instance] += loss.value()

                        loss.backward()
                        self.trainer.update()
                        dynet.renew_cg()  # use new computational graph for each BATCH when batching is active
                        batch = []
                else:
                    dynet.renew_cg() # new graph per item
                    output = self.predict(word_indices, char_indices, task_of_instance, train=True)
                    total_tagged += len(word_indices)

                    loss1 = dynet.esum([self.pick_neg_log(pred,gold) for pred, gold in zip(output, y)])
                    lv = loss1.value()
                    total_loss += lv

                    # logging
                    loss_accum_tagged[task_of_instance] += len(word_indices)
                    loss_accum_loss[task_of_instance] += loss1.value()

                    loss1.backward()
                    self.trainer.update()


            print("iter {2} {0:>12}: {1:.2f}".format("total loss",
                                                     total_loss/total_tagged,
                                                     iter), file=sys.stderr, flush=True)

            # log losses
            for task_id in sorted(losses):
                losses[task_id].append(loss_accum_loss[task_id] / loss_accum_tagged[task_id])

            if log_losses:
                pickle.dump(losses, open(model_path + ".model" + ".losses.pickle", "wb"))

            if dev:
                # evaluate after every epoch
                correct, total = self.evaluate(dev_X, dev_Y, org_X, org_Y, dev_task_labels)
                val_accuracy = correct/total
                print("\ndev accuracy: %.4f" % (val_accuracy),
                      file=sys.stderr, flush=True)

                if patience:
                    if val_accuracy > best_val_acc:
                        print('Accuracy %.4f is better than best val accuracy '
                              '%.4f.' % (val_accuracy, best_val_acc),
                              file=sys.stderr, flush=True)
                        best_val_acc = val_accuracy
                        epochs_no_improvement = 0
                        save(self, model_path)
                    else:
                        print('Accuracy %.4f is worse than best val loss %.4f.' %
                              (val_accuracy, best_val_acc), file=sys.stderr, flush=True)
                        epochs_no_improvement += 1
                    if epochs_no_improvement == patience:
                        print('No improvement for %d epochs. Early stopping...' %
                              epochs_no_improvement, file=sys.stderr, flush=True)
                        break


    def build_computation_graph(self, num_words, num_chars):
        """
        build graph and link to parameters
        """
        ## initialize word embeddings
        if self.embeds_file:
            print("loading embeddings", file=sys.stderr)
            embeddings, emb_dim = load_embeddings_file(self.embeds_file)
            assert(emb_dim==self.in_dim)
            num_words=len(set(embeddings.keys()).union(set(self.w2i.keys()))) # initialize all with embeddings
            # init model parameters and initialize them
            wembeds = self.model.add_lookup_parameters((num_words, self.in_dim), init=self.initializer)

            init=0
            for word in embeddings:
                if word not in self.w2i:
                    self.w2i[word]=len(self.w2i.keys()) # add new word
                    wembeds.init_row(self.w2i[word], embeddings[word])
                    init +=1 
                elif word in embeddings:
                    wembeds.init_row(self.w2i[word], embeddings[word])
                    init += 1
            print("initialized: {}".format(init), file=sys.stderr)

        else:
            wembeds = self.model.add_lookup_parameters((num_words, self.in_dim), init=self.initializer)


        ## initialize character embeddings
        cembeds = None
        if self.c_in_dim > 0:
            cembeds = self.model.add_lookup_parameters((num_chars, self.c_in_dim), init=self.initializer)
               

        # make it more flexible to add number of layers as specified by parameter
        layers = [] # inner layers
        output_layers_dict = {}   # from task_id to actual softmax predictor
        task_expected_at = {} # map task_id => output_layer_#

        # connect output layers to tasks
        for output_layer, task_id in zip(self.pred_layer, self.tasks_ids):
            if output_layer > self.h_layers:
                raise ValueError("cannot have a task at a layer (%d) which is "
                                 "beyond the model, increase h_layers (%d)"
                                 % (output_layer, self.h_layers))
            task_expected_at[task_id] = output_layer
        nb_tasks = len( self.tasks_ids )

        for layer_num in range(0,self.h_layers):
            if layer_num == 0:
                if self.c_in_dim > 0:
                    # in_dim: size of each layer
                    f_builder = self.builder(1, self.in_dim+self.c_in_dim*2, self.h_dim, self.model) 
                    b_builder = self.builder(1, self.in_dim+self.c_in_dim*2, self.h_dim, self.model) 
                else:
                    f_builder = self.builder(1, self.in_dim, self.h_dim, self.model)
                    b_builder = self.builder(1, self.in_dim, self.h_dim, self.model)

                layers.append(BiRNNSequencePredictor(f_builder, b_builder)) #returns forward and backward sequence
            else:
                # add inner layers (if h_layers >1)
                f_builder = self.builder(1, self.h_dim, self.h_dim, self.model)
                b_builder = self.builder(1, self.h_dim, self.h_dim, self.model)
                layers.append(BiRNNSequencePredictor(f_builder, b_builder))

        # store at which layer to predict task
        for task_id in self.tasks_ids:
            task_num_labels= len(self.task2tag2idx[task_id])
            output_layers_dict[task_id] = FFSequencePredictor(Layer(self.model, self.h_dim*2, task_num_labels, dynet.softmax, mlp=self.mlp, mlp_activation=self.activation_mlp))

        char_rnn = BiRNNSequencePredictor(self.builder(1, self.c_in_dim, self.c_in_dim, self.model),
                                          self.builder(1, self.c_in_dim, self.c_in_dim, self.model))

        predictors = {}
        predictors["inner"] = layers
        predictors["output_layers_dict"] = output_layers_dict
        predictors["task_expected_at"] = task_expected_at

        return predictors, char_rnn, wembeds, cembeds

    def get_features(self, words):
        """
        from a list of words, return the word and word char indices
        """
        word_indices = []
        word_char_indices = []
        for word in words:
            if word in self.w2i:
                word_indices.append(self.w2i[word])
            else:
                word_indices.append(self.w2i[UNK])

            if self.c_in_dim > 0:
                chars_of_word = [self.c2i["<w>"]]
                for char in word:
                    if char in self.c2i:
                        chars_of_word.append(self.c2i[char])
                    else:
                        chars_of_word.append(self.c2i[UNK])
                chars_of_word.append(self.c2i["</w>"])
                word_char_indices.append(chars_of_word)
        return word_indices, word_char_indices
                                                                                                                                

    def get_data_as_indices(self, folder_name, task, raw=False):
        """
        X = list of (word_indices, word_char_indices)
        Y = list of tag indices
        """
        X, Y = [],[]
        org_X, org_Y = [], []
        task_labels = []
        for (words, tags) in read_conll_file(folder_name, raw=raw):
            word_indices, word_char_indices = self.get_features(words)
            tag_indices = [self.task2tag2idx[task].get(tag) for tag in tags]
            X.append((word_indices,word_char_indices))
            Y.append(tag_indices)
            org_X.append(words)
            org_Y.append(tags)
            task_labels.append( task )
        return X, Y, org_X, org_Y, task_labels


    def predict(self, word_indices, char_indices, task_id, train=False):
        """
        predict tags for a sentence represented as char+word embeddings
        """

        # word embeddings
        wfeatures = [self.wembeds[w] for w in word_indices]

        # char embeddings
        if self.c_in_dim > 0:
            char_emb = []
            rev_char_emb = []
            # get representation for words
            for chars_of_token in char_indices:
                char_feats = [self.cembeds[c] for c in chars_of_token]
                # use last state as word representation
                f_char, b_char = self.char_rnn.predict_sequence(char_feats, char_feats)
                last_state = f_char[-1]
                rev_last_state = b_char[-1]
                char_emb.append(last_state)
                rev_char_emb.append(rev_last_state)

            features = [dynet.concatenate([w,c,rev_c]) for w,c,rev_c in zip(wfeatures,char_emb,rev_char_emb)]
        else:
            features = wfeatures
        
        if train: # only do at training time
            features = [dynet.noise(fe,self.noise_sigma) for fe in features]

        output_expected_at_layer = self.predictors["task_expected_at"][task_id]
        output_expected_at_layer -=1

        # go through layers
        # input is now combination of w + char emb
        prev = features
        prev_rev = features
        num_layers = self.h_layers

        for i in range(0,num_layers):
            predictor = self.predictors["inner"][i]
            forward_sequence, backward_sequence = predictor.predict_sequence(prev, prev_rev)        
            if i > 0 and self.activation:
                # activation between LSTM layers
                forward_sequence = [self.activation(s) for s in forward_sequence]
                backward_sequence = [self.activation(s) for s in backward_sequence]

            if i == output_expected_at_layer:
                output_predictor = self.predictors["output_layers_dict"][task_id] 
                concat_layer = [dynet.concatenate([f, b]) for f, b in zip(forward_sequence,reversed(backward_sequence))]

                if train and self.noise_sigma > 0.0:
                    concat_layer = [dynet.noise(fe,self.noise_sigma) for fe in concat_layer]
                output = output_predictor.predict_sequence(concat_layer)
                return output

            prev = forward_sequence
            prev_rev = backward_sequence 

        raise Exception("oops should not be here")
        return None

    def evaluate(self, test_X, test_Y, org_X, org_Y, task_labels, output_predictions=None, output_probs=False, verbose=True, raw=False):
        """
        compute accuracy on a test file
        """
        correct = 0
        total = 0.0

        if output_predictions != None:
            i2w = {self.w2i[w] : w for w in self.w2i.keys()}
            task_id = task_labels[0] # get first
            i2t = {self.task2tag2idx[task_id][t] : t for t in self.task2tag2idx[task_id].keys()}

        for i, ((word_indices, word_char_indices), gold_tag_indices, task_of_instance) in enumerate(zip(test_X, test_Y, task_labels)):
            if verbose:
                if i%100==0:
                    sys.stderr.write('%s'%i)
                elif i%10==0:
                    sys.stderr.write('.')

            output = self.predict(word_indices, word_char_indices, task_of_instance)
            predicted_tag_indices = [np.argmax(o.value()) for o in output]  # logprobs to indices
            if output_predictions:
                prediction = [i2t[idx] for idx in predicted_tag_indices]
                tag_confidences = [np.max(o.value()) for o in output]

                words = org_X[i]
                gold = org_Y[i]

                for w, g, p, c in zip(words, gold, prediction, tag_confidences):
                    if raw:
                        print(u"{}\t{}".format(w, p)) # do not print DUMMY tag when --raw is on
                    else:
                        if output_probs:
                            print(u"%s\t%s\t%s\t%.2f" % (w, g, p, c))
                        else:
                            print(u"%s\t%s\t%s" % (w, g, p))
                print("")

            correct += sum([1 for (predicted, gold) in zip(predicted_tag_indices, gold_tag_indices) if predicted == gold])
            total += len(gold_tag_indices)

        return correct, total


    def get_train_data(self, list_folders_name):
        """
        Get train data: read each train set (linked to a task)

        :param list_folders_name: list of folders names

        transform training data to features (word indices)
        map tags to integers
        """
        X = []
        Y = []
        task_labels = [] # keeps track of where instances come from "task1" or "task2"..
        self.tasks_ids = [] # record ids of the tasks

        # word 2 indices and tag 2 indices
        w2i = {} # word to index
        c2i = {} # char to index
        task2tag2idx = {} # id of the task -> tag2idx

        w2i[UNK] = 0  # unk word / OOV
        c2i[UNK] = 0  # unk char
        c2i["<w>"] = 1   # word start
        c2i["</w>"] = 2  # word end index

        if self.max_vocab_size is not None:
            word_counter = Counter()
            print('Reading files to create vocabulary of size %d.' %
                  self.max_vocab_size)
            for i, folder_name in enumerate(list_folders_name):
                for words, _ in read_conll_file(folder_name):
                    word_counter.update(words)
            word_count_pairs = word_counter.most_common(self.max_vocab_size-1)
            for word, _ in word_count_pairs:
                w2i[word] = len(w2i)

        for i, folder_name in enumerate(list_folders_name):
            num_sentences=0
            num_tokens=0
            task_id = 'task'+str(i)
            self.tasks_ids.append( task_id )
            if task_id not in task2tag2idx:
                task2tag2idx[task_id] = {}
            for instance_idx, (words, tags) in enumerate(read_conll_file(folder_name)):
                num_sentences += 1
                instance_word_indices = [] #sequence of word indices
                instance_char_indices = [] #sequence of char indices 
                instance_tags_indices = [] #sequence of tag indices

                for i, (word, tag) in enumerate(zip(words, tags)):
                    num_tokens += 1

                    # map words and tags to indices
                    if word not in w2i and self.max_vocab_size is not None:
                        # if word is not in the created vocab, add an UNK token
                        instance_word_indices.append(w2i[UNK])
                    else:
                        if word not in w2i:
                            w2i[word] = len(w2i)
                        instance_word_indices.append(w2i[word])

                    if self.c_in_dim > 0:
                        chars_of_word = [c2i["<w>"]]
                        for char in word:
                            if char not in c2i:
                                c2i[char] = len(c2i)
                            chars_of_word.append(c2i[char])
                        chars_of_word.append(c2i["</w>"])
                        instance_char_indices.append(chars_of_word)
                            
                    if tag not in task2tag2idx[task_id]:
                        task2tag2idx[task_id][tag]=len(task2tag2idx[task_id])

                    instance_tags_indices.append(task2tag2idx[task_id].get(tag))

                X.append((instance_word_indices, instance_char_indices)) # list of word indices, for every word list of char indices
                Y.append(instance_tags_indices)
                task_labels.append(task_id)

            if num_sentences == 0 or num_tokens == 0:
                sys.exit( "No data read from: "+folder_name )

            print("TASK "+task_id+" "+folder_name, file=sys.stderr )
            print("%s sentences %s tokens" % (num_sentences, num_tokens), file=sys.stderr)
            print("%s w features, %s c features " % (len(w2i),len(c2i)), file=sys.stderr)

        assert(len(X)==len(Y))
        return X, Y, task_labels, w2i, c2i, task2tag2idx  #sequence of features, sequence of labels, necessary mappings


    def save_embeds(self, out_filename):
        """
        save final embeddings to file
        :param out_filename: filename
        """
        # construct reverse mapping
        i2w = {self.w2i[w]: w for w in self.w2i.keys()}

        OUT = open(out_filename+".w.emb","w")
        for word_id in i2w.keys():
            wembeds_expression = self.wembeds[word_id]
            word = i2w[word_id]
            OUT.write("{} {}\n".format(word," ".join([str(x) for x in wembeds_expression.npvalue()])))
        OUT.close()


if __name__=="__main__":
    main()
