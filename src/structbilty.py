#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DSDS - a neural network based tagger (bi-LSTM) - re-factored tagger from https://arxiv.org/abs/1604.05529 with support for bilstm-CRF
:author: Barbara Plank
"""
import argparse
import array
import random
import time
import sys
import numpy as np
import os
import dill
import _dynet as dynet
import codecs

from collections import Counter, defaultdict
from lib.mnnl import FFSequencePredictor, Layer, BiRNNSequencePredictor, CRFSequencePredictor, init_dynet, is_in_dict
from lib.mio import load_embeddings_file, SeqData, load_dict
from lib.constants import UNK, MAX_SEED, WORD_START, WORD_END, START_TAG, END_TAG
from lib.mmappers import TRAINER_MAP, ACTIVATION_MAP, INITIALIZER_MAP, BUILDERS

def main():
    parser = argparse.ArgumentParser(description="""Run the bi-LSTM tagger""", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    group_main = parser.add_argument_group('Main', 'main arguments')
    group_main.add_argument("--model", help="path to store/load model [required]", required=True)
    group_main.add_argument("--train", nargs='*', help="path to train file [if multiple files are given actives MTL]") # allow multiple train files, each asociated with a task = position in the list
    group_main.add_argument("--dev", nargs='*', help="dev file(s)", required=False)
    group_main.add_argument("--test", nargs='*', help="test file(s) [same order as --train]", required=False)

    group_model = parser.add_argument_group('Model', 'specify model parameters')
    group_model.add_argument("--in_dim", help="input dimension", type=int, default=64) # default Polyglot size
    group_model.add_argument("--h_dim", help="hidden dimension [default: 100]", type=int, default=100)
    group_model.add_argument("--c_in_dim", help="input dimension for character embeddings", type=int, default=100)
    group_model.add_argument("--c_h_dim", help="hidden dimension for character embeddings", type=int, default=100)
    group_model.add_argument("--h_layers", help="number of stacked LSTMs [default: 1 = no stacking]", required=False, type=int, default=1)
    group_model.add_argument("--pred_layer", nargs='*', help="predict task at this layer [default: last layer]", required=False) # for each task the layer on which it is predicted (default 1)
    group_model.add_argument("--embeds", help="word embeddings file", required=False, default=None)
    group_model.add_argument("--crf", help="use CRF instead of local decoding", default=False, action="store_true")
    group_model.add_argument("--viterbi-loss", help="Use viterbi loss training (only active if --crf is on)", action="store_true", default=False)
    group_model.add_argument("--transition-matrix", help="store transition matrix from CRF")

    group_model.add_argument("--builder", help="RNN builder (default: lstmc)", choices=BUILDERS.keys(), default="lstmc")

    group_model.add_argument("--mlp", help="add additional MLP layer of this dimension [default 0=disabled]", default=0, type=int)
    group_model.add_argument("--ac-mlp", help="activation function for optional MLP layer [rectify, tanh, ...] (default: tanh)",
                        default="tanh", choices=ACTIVATION_MAP.keys())
    group_model.add_argument("--ac", help="activation function between hidden layers [rectify, tanh, ...]", default="tanh",
                             choices=ACTIVATION_MAP.keys())

    group_input = parser.add_argument_group('Input', 'specific input options')
    group_input.add_argument("--raw", help="expects raw text input (one sentence per line)", required=False, action="store_true", default=False)

    group_output = parser.add_argument_group('Output', 'specific output options')
    group_output.add_argument("--dictionary", help="use dictionary as additional features or type constraints (with --type-constraints)", default=None)
    group_output.add_argument("--type-constraint", help="use dictionary as type constraints", default=False, action="store_true")
    group_output.add_argument("--embed-lex", help="use dictionary as type constraints", default=False, action="store_true")
    group_output.add_argument("--lex-dim", help="input dimension for lexical features", default=0, type=int)
    group_output.add_argument("--output", help="output predictions to file [word|gold|pred]", default=None)
    group_output.add_argument("--output-confidences", help="output tag confidences", action="store_true", default=False)
    group_output.add_argument("--save-embeds", help="save word embeddings to file", required=False, default=None)
    group_output.add_argument("--save-lexembeds", help="save lexicon embeddings to file", required=False, default=None)
    group_output.add_argument("--save-cwembeds", help="save character-based word-embeddings to file", required=False, default=None)
    group_output.add_argument("--save-lwembeds", help="save lexicon-based word-embeddings to file", required=False, default=None)
    group_output.add_argument("--mimickx-model", help="use mimickx model for OOVs", required=False, default=None, type=str)


    group_opt = parser.add_argument_group('Optimizer', 'specify training parameters')
    group_opt.add_argument("--iters", help="training iterations", type=int,default=20)
    group_opt.add_argument("--sigma", help="sigma of Gaussian noise",default=0.2, type=float)
    group_opt.add_argument("--trainer", help="trainer [default: sgd]", choices=TRAINER_MAP.keys(), default="sgd")
    group_opt.add_argument("--learning-rate", help="learning rate [0: use default]", default=0, type=float) # see: http://dynet.readthedocs.io/en/latest/optimizers.html
    group_opt.add_argument("--patience", help="patience [default: 0=not used], requires specification of --dev and model path --save", required=False, default=0, type=int)
    group_opt.add_argument("--log-losses", help="log loss (for each task if multiple active)", required=False, action="store_true", default=False)
    group_opt.add_argument("--word-dropout-rate", help="word dropout rate [default: 0.25], if 0=disabled, recommended: 0.25 (Kiperwasser & Goldberg, 2016)", required=False, default=0.25, type=float)
    group_opt.add_argument("--char-dropout-rate", help="char dropout rate [default: 0=disabled]", required=False, default=0.0, type=float)
    group_opt.add_argument("--disable-backprob-embeds", help="disable backprob into embeddings (default is to update)",
                        required=False, action="store_false", default=True)
    group_opt.add_argument("--initializer", help="initializer for embeddings (default: constant)",
                        choices=INITIALIZER_MAP.keys(), default="constant")


    group_dynet = parser.add_argument_group('DyNet', 'DyNet parameters')
    group_dynet.add_argument("--seed", help="random seed (also for DyNet)", required=False, type=int)
    group_dynet.add_argument("--dynet-mem", help="memory for DyNet", required=False, type=int)
    group_dynet.add_argument("--dynet-gpus", help="1 for GPU usage", default=0, type=int) # warning: non-deterministic results on GPU https://github.com/clab/dynet/issues/399
    group_dynet.add_argument("--dynet-autobatch", help="if 1 enable autobatching", default=0, type=int)
    group_dynet.add_argument("--minibatch-size", help="size of minibatch for autobatching (1=disabled)", default=1, type=int)

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        exit()

    if args.train:
        if len(args.train) > 1:
            if not args.pred_layer:
                print("--pred_layer required!")
                exit()
        elif len(args.train) == 1 and not args.pred_layer:
            args.pred_layer = [args.h_layers] # assumes h_layers is 1

    if args.c_in_dim == 0:
        print(">>> disable character embeddings <<<")

    if args.minibatch_size > 1:
        print(">>> using minibatch_size {} <<<".format(args.minibatch_size))

    if args.viterbi_loss:
        if not args.crf:
            print("--crf (global decoding) needs to be active when --viterbi is used")
            exit()
    if args.crf:
        if args.viterbi_loss:
            print(">>> using global decoding (Viterbi loss) <<<")
        else:
            print(">>> using global decoding (CRF, neg-log loss) <<<")

    if args.patience:
        if not args.dev or not args.model:
            print("patience requires a dev set and model path (--dev and --model)")
            exit()

    # check if --save folder exists
    if args.model:
        if os.path.isdir(args.model):
            if not os.path.exists(args.model):
                print("Creating {}..".format(args.model))
                os.makedirs(args.model)
        elif os.path.isdir(os.path.dirname(args.model)) and not os.path.exists(os.path.dirname(args.model)):
            print("Creating {}..".format(os.path.dirname(args.model)))
            os.makedirs(os.path.dirname(args.model))

    if args.output:
        if os.path.isdir(os.path.dirname(args.output)) and not os.path.exists(os.path.dirname(args.output)):
            os.makedirs(os.path.dirname(args.output))

    if not args.seed:
        ## set seed
        seed = random.randint(1, MAX_SEED)
    else:
        seed = args.seed

    print(">>> using seed: {} <<< ".format(seed))
    np.random.seed(seed)
    random.seed(seed)

    init_dynet(seed)

    if args.mimickx_model:
        from mimickx import Mimickx, load_model  # make sure PYTHONPATH is set
        print(">>> Loading mimickx model {} <<<".format(args.mimickx_model))

    model_path = args.model

    start = time.time()

    if args.train and len( args.train ) != 0:

        tagger = NNTagger(args.in_dim,
                          args.h_dim,
                          args.c_in_dim,
                          args.c_h_dim,
                          args.h_layers,
                          args.pred_layer,
                          embeds_file=args.embeds,
                          w_dropout_rate=args.word_dropout_rate,
                          c_dropout_rate=args.char_dropout_rate,
                          activation=ACTIVATION_MAP[args.ac],
                          mlp=args.mlp,
                          activation_mlp=ACTIVATION_MAP[args.ac_mlp],
                          noise_sigma=args.sigma,
                          learning_algo=args.trainer,
                          learning_rate=args.learning_rate,
                          backprob_embeds=args.disable_backprob_embeds,
                          initializer=INITIALIZER_MAP[args.initializer],
                          builder=BUILDERS[args.builder],
                          crf=args.crf,
                          mimickx_model_path=args.mimickx_model,
                          dictionary=args.dictionary, type_constraint=args.type_constraint,
                          lex_dim=args.lex_dim, embed_lex=args.embed_lex)

        dev = None
        train = SeqData(args.train)
        if args.dev:
            dev = SeqData(args.dev)

        tagger.fit(train, args.iters,
                   dev=dev,
                   model_path=model_path, patience=args.patience, minibatch_size=args.minibatch_size, log_losses=args.log_losses)

        if not args.dev and not args.patience:  # in case patience is active it gets saved in the fit function
            save(tagger, model_path)

    if args.test and len( args.test ) != 0:

        tagger = load(args.model, args.dictionary)

        # check if mimickx provided after training
        if args.mimickx_model:
            tagger.mimickx_model_path = args.mimickx_model
            tagger.mimickx_model = load_model(args.mimickx_model)

        stdout = sys.stdout
        # One file per test ...
        if args.test:
            test = SeqData(args.test) # read in all test data

            for i, test_file in enumerate(args.test): # expect them in same order
                if args.output is not None:
                    sys.stdout = codecs.open(args.output + ".task{}".format(i), 'w', encoding='utf-8')

                start_testing = time.time()

                print('\nTesting task{}'.format(i),file=sys.stderr)
                print('*******\n',file=sys.stderr)
                correct, total = tagger.evaluate(test, "task{}".format(i),
                                                 output_predictions=args.output,
                                                 output_confidences=args.output_confidences, raw=args.raw,
                                                 unk_tag=None)
                if not args.raw:
                    print("\nTask{} test accuracy on {} items: {:.4f}".format(i, i+1, correct/total),file=sys.stderr)
                print(("Done. Took {0:.2f} seconds in total (testing took {1:.2f} seconds).".format(time.time()-start,
                                                                                                    time.time()-start_testing)),file=sys.stderr)
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

    tagger = load(args.model, args.dictionary)

    if args.save_embeds:
        tagger.save_embeds(args.save_embeds)

    if args.save_lexembeds:
        tagger.save_lex_embeds(args.save_lexembeds)

    if args.save_cwembeds:
        tagger.save_cw_embeds(args.save_cwembeds)

    if args.save_lwembeds:
        tagger.save_lw_embeds(args.save_lwembeds)
    
    if args.transition_matrix:
        tagger.save_transition_matrix(args.transition_matrix)




def load(model_path, local_dictionary=None):
    """
    load a model from file; specify the .model file, it assumes the *pickle file in the same location
    """
    print("load model.. ", model_path)
    myparams = dill.load(open(model_path+".params.pickle", "rb"))
    if not "mimickx_model_path" in myparams:
        myparams["mimickx_model_path"] = None
    if local_dictionary:
        myparams["path_to_dictionary"] = local_dictionary
    tagger = NNTagger(myparams["in_dim"],
                      myparams["h_dim"],
                      myparams["c_in_dim"],
                      myparams["c_h_dim"],
                      myparams["h_layers"],
                      myparams["pred_layer"],
                      activation=myparams["activation"],
                      mlp=myparams["mlp"],
                      activation_mlp=myparams["activation_mlp"],
                      builder=myparams["builder"],
                      crf=myparams["crf"],
                      mimickx_model_path=myparams["mimickx_model_path"],
                      dictionary=myparams["path_to_dictionary"],
                      type_constraint=myparams["type_constraint"],
                      lex_dim=myparams["lex_dim"],
                      embed_lex=myparams["embed_lex"]
                      )
    tagger.set_indices(myparams["w2i"],myparams["c2i"],myparams["task2tag2idx"],myparams["w2c_cache"], myparams["l2i"])
    tagger.set_counts(myparams["wcount"], myparams["wtotal"], myparams["ccount"], myparams["ctotal"])
    tagger.build_computation_graph(myparams["num_words"],
                                       myparams["num_chars"])
    tagger.model.populate(model_path+".model")
    print("model loaded: {}".format(model_path))
    return tagger


def save(nntagger, model_path):
    """
    save a model; dynet only saves the parameters, need to store the rest separately
    """
    modelname = model_path + ".model"
    nntagger.model.save(modelname)
    myparams = {"num_words": len(nntagger.w2i),
                "num_chars": len(nntagger.c2i),
                "w2i": nntagger.w2i,
                "c2i": nntagger.c2i,
                "wcount": nntagger.wcount,
                "wtotal": nntagger.wtotal,
                "ccount": nntagger.ccount,
                "ctotal": nntagger.ctotal,
                "w2c_cache": nntagger.w2c_cache,
                "task2tag2idx": nntagger.task2tag2idx,
                "activation": nntagger.activation,
                "mlp": nntagger.mlp,
                "activation_mlp": nntagger.activation_mlp,
                "in_dim": nntagger.in_dim,
                "h_dim": nntagger.h_dim,
                "c_in_dim": nntagger.c_in_dim,
                "c_h_dim": nntagger.c_h_dim,
                "h_layers": nntagger.h_layers,
                "pred_layer": nntagger.pred_layer,
                "builder": nntagger.builder,
                "crf": nntagger.crf,
                "mimickx_model_path": nntagger.mimickx_model_path,
                "path_to_dictionary": nntagger.path_to_dictionary,
                "type_constraint": nntagger.type_constraint,
                "lex_dim": nntagger.lex_dim,
                "embed_lex": nntagger.embed_lex,
                "l2i": nntagger.l2i
                }
    dill.dump(myparams, open( model_path+".params.pickle", "wb" ) )
    print("model stored: {}".format(modelname))
    del nntagger


def drop(x, xcount, dropout_rate):
    """
    drop x if x is less frequent (cf. Kiperwasser & Goldberg, 2016)
    """
    return random.random() > (xcount.get(x)/(dropout_rate+xcount.get(x)))


class NNTagger(object):

    # turn dynamic allocation off by defining slots
    __slots__ = ['w2i', 'c2i', 'wcount', 'ccount','wtotal','ctotal','w2c_cache','w_dropout_rate','c_dropout_rate',
                  'task2tag2idx', 'model', 'in_dim', 'c_in_dim', 'c_h_dim','h_dim', 'activation',
                 'noise_sigma', 'pred_layer', 'mlp', 'activation_mlp', 'backprob_embeds', 'initializer',
                 'h_layers', 'predictors', 'wembeds', 'cembeds', 'embeds_file', 'char_rnn', 'trainer',
                 'builder', 'crf', 'viterbi_loss', 'mimickx_model_path', 'mimickx_model',
                 'dictionary',  'dictionary_values', 'path_to_dictionary', 'lex_dim', 'type_constraint',
                 'embed_lex', 'l2i', 'lembeds']

    def __init__(self,in_dim,h_dim,c_in_dim,c_h_dim,h_layers,pred_layer,learning_algo="sgd", learning_rate=0,
                 embeds_file=None,activation=ACTIVATION_MAP["tanh"],mlp=0,activation_mlp=ACTIVATION_MAP["rectify"],
                 backprob_embeds=True,noise_sigma=0.1, w_dropout_rate=0.25, c_dropout_rate=0.25,
                 initializer=INITIALIZER_MAP["glorot"], builder=BUILDERS["lstmc"], crf=False, viterbi_loss=False,
                 mimickx_model_path=None, dictionary=None, type_constraint=False,
                 lex_dim=0, embed_lex=False):
        self.w2i = {}  # word to index mapping
        self.c2i = {}  # char to index mapping
        self.w2c_cache = {} # word to char index cache for frequent words
        self.wcount = None # word count
        self.ccount = None # char count
        self.task2tag2idx = {} # need one dictionary per task
        self.pred_layer = [int(layer) for layer in pred_layer] # at which layer to predict each task
        self.model = dynet.ParameterCollection() #init model
        self.in_dim = in_dim
        self.h_dim = h_dim
        self.c_in_dim = c_in_dim
        self.c_h_dim = c_h_dim
        self.w_dropout_rate = w_dropout_rate
        self.c_dropout_rate = c_dropout_rate
        self.activation = activation
        self.mlp = mlp
        self.activation_mlp = activation_mlp
        self.noise_sigma = noise_sigma
        self.h_layers = h_layers
        self.predictors = {"inner": [], "output_layers_dict": {}, "task_expected_at": {} } # the inner layers and predictors
        self.wembeds = None # lookup: embeddings for words
        self.cembeds = None # lookup: embeddings for characters
        self.lembeds = None # lookup: embeddings for lexical features (optional)
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
        self.crf = crf
        self.viterbi_loss = viterbi_loss
        self.mimickx_model_path = mimickx_model_path
        if mimickx_model_path: # load
            self.mimickx_model = load_model(mimickx_model_path)
        self.dictionary = None
        self.type_constraint = type_constraint
        self.embed_lex = False
        self.l2i = {UNK: 0}  # lex feature to index mapping
        if dictionary:
            self.dictionary, self.dictionary_values = load_dict(dictionary)
            self.path_to_dictionary = dictionary
            if type_constraint:
                self.lex_dim = 0
            else:
                if embed_lex:
                    self.lex_dim = lex_dim
                    self.embed_lex = True
                    print("Embed lexical features")
                    # register property indices
                    for prop in self.dictionary_values:
                        self.l2i[prop] = len(self.l2i)
                else:
                    self.lex_dim = len(self.dictionary_values) #n-hot encoding
                print("Lex_dim: {}".format(self.lex_dim), file=sys.stderr)
        else:
            self.dictionary = None
            self.path_to_dictionary = None
            self.lex_dim = 0

    def fit(self, train, num_iterations, dev=None, model_path=None, patience=0, minibatch_size=0, log_losses=False):
        """
        train the tagger
        """
        losses_log = {} # log losses

        print("init parameters")
        self.init_parameters(train)

        # init lookup parameters and define graph
        print("build graph")
        self.build_computation_graph(len(self.w2i),  len(self.c2i))

        update_embeds = True
        if self.backprob_embeds == False: ## disable backprob into embeds
            print(">>> disable wembeds update <<<")
            update_embeds = False
            
        best_val_acc, epochs_no_improvement = 0.0, 0

        if dev and model_path is not None and patience > 0:
            print('Using early stopping with patience of {}...'.format(patience))

        batch = []
        print("train..")
        for iteration in range(num_iterations):

            total_loss=0.0
            total_tagged=0.0

            indices = [i for i in range(len(train.seqs))]
            random.shuffle(indices)

            loss_accum_loss = defaultdict(float)
            loss_accum_tagged = defaultdict(float)

            for idx in indices:
                seq = train.seqs[idx]

                if seq.task_id not in losses_log:
                    losses_log[seq.task_id] = [] #initialize

                if minibatch_size > 1:
                    # accumulate instances for minibatch update
                    loss1 = self.predict(seq, train=True, update_embeds=update_embeds)
                    total_tagged += len(seq.words)
                    batch.append(loss1)
                    if len(batch) == minibatch_size:
                        loss = dynet.esum(batch)
                        total_loss += loss.value()

                        # logging
                        loss_accum_tagged[seq.task_id] += len(seq.words)
                        loss_accum_loss[seq.task_id] += loss.value()

                        loss.backward()
                        self.trainer.update()
                        dynet.renew_cg()  # use new computational graph for each BATCH when batching is active
                        batch = []
                else:
                    dynet.renew_cg() # new graph per item
                    loss1 = self.predict(seq, train=True, update_embeds=update_embeds)
                    total_tagged += len(seq.words)
                    lv = loss1.value()
                    total_loss += lv

                    # logging
                    loss_accum_tagged[seq.task_id] += len(seq.words)
                    loss_accum_loss[seq.task_id] += loss1.value()

                    loss1.backward()
                    self.trainer.update()

            print("iter {2} {0:>12}: {1:.2f}".format("total loss", total_loss/total_tagged, iteration))

            # log losses
            for task_id in sorted(losses_log):
                losses_log[task_id].append(loss_accum_loss[task_id] / loss_accum_tagged[task_id])

            if log_losses:
                dill.dump(losses_log, open(model_path + ".model" + ".losses.pickle", "wb"))

            if dev:
                # evaluate after every epoch
                correct, total = self.evaluate(dev, "task0")
                val_accuracy = correct/total
                print("dev accuracy: {0:.4f}".format(val_accuracy))

                if val_accuracy > best_val_acc:
                    print('Accuracy {0:.4f} is better than best val accuracy '
                          '{1:.4f}.'.format(val_accuracy, best_val_acc))
                    best_val_acc = val_accuracy
                    epochs_no_improvement = 0
                    save(self, model_path)
                else:
                    print('Accuracy {0:.4f} is worse than best val loss {1:.4f}.'.format(val_accuracy, best_val_acc))
                    epochs_no_improvement += 1

                if patience > 0:
                    if epochs_no_improvement == patience:
                        print('No improvement for {} epochs. Early stopping...'.format(epochs_no_improvement))
                        break

    def set_indices(self, w2i, c2i, task2t2i, w2c_cache, l2i=None):
        """ helper function for loading model"""
        for task_id in task2t2i:
            self.task2tag2idx[task_id] = task2t2i[task_id]
        self.w2i = w2i
        self.c2i = c2i
        self.w2c_cache = w2c_cache
        self.l2i = l2i

    def set_counts(self, wcount, wtotal, ccount, ctotal):
        """ helper function for loading model"""
        self.wcount = wcount
        self.wtotal = wtotal
        self.ccount = ccount
        self.ctotal = ctotal

    def build_computation_graph(self, num_words, num_chars):
        """
        build graph and link to parameters
        self.predictors, self.char_rnn, self.wembeds, self.cembeds =
        """
        ## initialize word embeddings
        if self.embeds_file:
            print("loading embeddings")
            embeddings, emb_dim = load_embeddings_file(self.embeds_file)
            assert(emb_dim==self.in_dim)
            num_words=len(set(embeddings.keys()).union(set(self.w2i.keys()))) # initialize all with embeddings
            # init model parameters and initialize them
            self.wembeds = self.model.add_lookup_parameters((num_words, self.in_dim), init=self.initializer)

            init=0
            for word in embeddings.keys():
                if word not in self.w2i:
                    self.w2i[word]=len(self.w2i.keys()) # add new word
                    self.wembeds.init_row(self.w2i[word], embeddings[word])
                    init +=1
                elif word in embeddings:
                    self.wembeds.init_row(self.w2i[word], embeddings[word])
                    init += 1
            print("initialized: {}".format(init))
            del embeddings # clean up
        else:
            self.wembeds = self.model.add_lookup_parameters((num_words, self.in_dim), init=self.initializer)

        ## initialize character embeddings
        self.cembeds = None
        if self.c_in_dim > 0:
            self.cembeds = self.model.add_lookup_parameters((num_chars, self.c_in_dim), init=self.initializer)
        if self.lex_dim > 0 and self.embed_lex:
            # +1 for UNK property
            self.lembeds = self.model.add_lookup_parameters((len(self.dictionary_values)+1, self.lex_dim), init=dynet.GlorotInitializer()) #init=self.initializer)

        # make it more flexible to add number of layers as specified by parameter
        layers = [] # inner layers
        output_layers_dict = {}   # from task_id to actual softmax predictor
        for layer_num in range(0,self.h_layers):
            if layer_num == 0:
                if self.c_in_dim > 0:
                    # in_dim: size of each layer
                    if self.lex_dim > 0 and self.embed_lex:
                        lex_embed_size = self.lex_dim * len(self.dictionary_values)
                        f_builder = self.builder(1, self.in_dim+self.c_h_dim*2+lex_embed_size, self.h_dim, self.model)
                        b_builder = self.builder(1, self.in_dim+self.c_h_dim*2+lex_embed_size, self.h_dim, self.model)
                    else:
                        f_builder = self.builder(1, self.in_dim + self.c_h_dim * 2 + self.lex_dim, self.h_dim, self.model)
                        b_builder = self.builder(1, self.in_dim + self.c_h_dim * 2 + self.lex_dim, self.h_dim, self.model)
                else:
                    f_builder = self.builder(1, self.in_dim+self.lex_dim, self.h_dim, self.model)
                    b_builder = self.builder(1, self.in_dim+self.lex_dim, self.h_dim, self.model)

                layers.append(BiRNNSequencePredictor(f_builder, b_builder)) #returns forward and backward sequence
            else:
                # add inner layers (if h_layers >1)
                f_builder = self.builder(1, self.h_dim, self.h_dim, self.model)
                b_builder = self.builder(1, self.h_dim, self.h_dim, self.model)
                layers.append(BiRNNSequencePredictor(f_builder, b_builder))

        # store at which layer to predict task
        task2layer = {task_id: out_layer for task_id, out_layer in zip(self.task2tag2idx, self.pred_layer)}
        if len(task2layer) > 1:
            print("task2layer", task2layer)
        for task_id in task2layer:
            task_num_labels= len(self.task2tag2idx[task_id])
            if not self.crf:
                output_layers_dict[task_id] = FFSequencePredictor(self.task2tag2idx[task_id], Layer(self.model, self.h_dim*2, task_num_labels,
                                                                                                    dynet.softmax, mlp=self.mlp, mlp_activation=self.activation_mlp))
            else:
                print("CRF")
                output_layers_dict[task_id] = CRFSequencePredictor(self.model, task_num_labels,
                                                                   self.task2tag2idx[task_id],
                                                                   Layer(self.model, self.h_dim * 2, task_num_labels,
                                                                        None, mlp=self.mlp,
                                                                        mlp_activation=self.activation_mlp), viterbi_loss=self.viterbi_loss)

        self.char_rnn = BiRNNSequencePredictor(self.builder(1, self.c_in_dim, self.c_h_dim, self.model),
                                          self.builder(1, self.c_in_dim, self.c_h_dim, self.model))

        self.predictors = {}
        self.predictors["inner"] = layers
        self.predictors["output_layers_dict"] = output_layers_dict
        self.predictors["task_expected_at"] = task2layer

        
    def get_features(self, words, train=False, update=True):
        """
        get feature representations
        """
        # word embeddings
        wfeatures = np.array([self.get_w_repr(word, train=train, update=update) for word in words])

        lex_features = []
        if self.dictionary and not self.type_constraint:
            ## add lexicon features
            lex_features = np.array([self.get_lex_repr(word) for word in words])
        # char embeddings
        if self.c_in_dim > 0:
            cfeatures = [self.get_c_repr(word, train=train) for word in words]
            if len(lex_features) > 0:
                lex_features = dynet.inputTensor(lex_features)
                features = [dynet.concatenate([w,c,l]) for w,c,l in zip(wfeatures,cfeatures,lex_features)]
            else:
                features = [dynet.concatenate([w, c]) for w, c in zip(wfeatures, cfeatures)]
        else:
            features = wfeatures
        if train: # only do at training time
            features = [dynet.noise(fe,self.noise_sigma) for fe in features]
        return features

    def predict(self, seq, train=False, output_confidences=False, unk_tag=None, update_embeds=True):
        """
        predict tags for a sentence represented as char+word embeddings and compute losses for this instance
        """
        if not train:
            dynet.renew_cg()
        features = self.get_features(seq.words, train=train, update=update_embeds)

        output_expected_at_layer = self.predictors["task_expected_at"][seq.task_id]
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
                output_predictor = self.predictors["output_layers_dict"][seq.task_id]
                concat_layer = [dynet.concatenate([f, b]) for f, b in zip(forward_sequence,reversed(backward_sequence))]

                if train and self.noise_sigma > 0.0:
                    concat_layer = [dynet.noise(fe,self.noise_sigma) for fe in concat_layer]
                # fill-in predictions and get loss per tag
                losses = output_predictor.predict_sequence(seq, concat_layer,
                                                           train=train, output_confidences=output_confidences,
                                                           unk_tag=unk_tag, dictionary=self.dictionary,
                                                           type_constraint=self.type_constraint)

            prev = forward_sequence
            prev_rev = backward_sequence 

        if train:
            # return losses
            return losses
        else:
            return seq.pred_tags, seq.tag_confidences

    def output_preds(self, seq, raw=False, output_confidences=False):
        """
        output predictions to a file
        """
        i = 0
        for w, g, p in zip(seq.words, seq.tags, seq.pred_tags):
            if raw:
                if output_confidences:
                    print(u"{0}\t{1}\t{2:.2f}".format(w, p, seq.tag_confidences[i]))
                else:
                    print(u"{}\t{}".format(w, p))  # do not print DUMMY tag when --raw is on
            else:
                if output_confidences:
                    print(u"{0}\t{1}\t{2}\t{3:.2f}".format(w, g, p, seq.tag_confidences[i]))
                else:
                    print(u"{}\t{}\t{}".format(w, g, p))
            i += 1
        print("")

    def evaluate(self, test_file, task_id, output_predictions=None, raw=False, output_confidences=False, unk_tag=None):
        """
        compute accuracy on a test file, optionally output to file
        """
        correct = 0
        total = 0

        for seq in test_file:
            if seq.task_id != task_id:
                continue # we evaluate only on a specific task
            self.predict(seq, output_confidences=output_confidences, unk_tag=unk_tag)
            if output_predictions:
                self.output_preds(seq, raw=raw, output_confidences=output_confidences)
            correct_inst, total_inst = seq.evaluate()
            correct+=correct_inst
            total+= total_inst
        return correct, total

    def get_w_repr(self, word, train=False, update=True):
        """
        Get representation of word (word embedding)
        """
        if train:
            if self.w_dropout_rate > 0.0:
                w_id = self.w2i[UNK] if drop(word, self.wcount, self.w_dropout_rate) else self.w2i.get(word, self.w2i[UNK])
        else:
            if self.mimickx_model_path: # if given use MIMICKX
                if word not in self.w2i: #
                    #print("predict with MIMICKX for: ", word)
                    return dynet.inputVector(self.mimickx_model.predict(word).npvalue())
            w_id = self.w2i.get(word, self.w2i[UNK])
        if not update:
            return dynet.nobackprop(self.wembeds[w_id])
        else:
            return self.wembeds[w_id] 

    def get_c_repr(self, word, train=False):
        """
        Get representation of word via characters sub-LSTMs
        """
        # get representation for words
        if word in self.w2c_cache:
            chars_of_token = self.w2c_cache[word]
            if train:
                chars_of_token = [drop(c, self.ccount, self.c_dropout_rate) for c in chars_of_token]
        else:
            chars_of_token = array.array('I',[self.c2i[WORD_START]]) + array.array('I',[self.get_c_idx(c, train=train) for c in word]) + array.array('I',[self.c2i[WORD_END]])

        char_feats = [self.cembeds[c_id] for c_id in chars_of_token]
        # use last state as word representation
        f_char, b_char = self.char_rnn.predict_sequence(char_feats, char_feats)
        return dynet.concatenate([f_char[-1], b_char[-1]])

    def get_c_idx(self, c, train=False):
        """ helper function to get index of character"""
        if self.c_dropout_rate > 0.0 and train and drop(c, self.ccount, self.c_dropout_rate):
            return self.c2i.get(UNK)
        else:
            return self.c2i.get(c, self.c2i[UNK])

    def get_lex_repr(self, word):
        """
        Get representation for lexical feature
        """
        if not self.embed_lex: ## n-hot representation
            n_hot = np.zeros(len(self.dictionary_values))
            values = is_in_dict(word, self.dictionary)
            if values:
                for v in values:
                    n_hot[self.dictionary_values.index(v)] = 1.0
            return n_hot
        else:
            lex_feats = []
            for property in self.dictionary_values:
                values = is_in_dict(word, self.dictionary)
                if values:
                    if property in values:
                        lex_feats.append(self.lembeds[self.l2i[property]].npvalue())
                    else:
                        lex_feats.append(self.lembeds[self.l2i[UNK]].npvalue())
                else:
                    lex_feats.append(self.lembeds[self.l2i[UNK]].npvalue()) # unknown word
            return np.concatenate(lex_feats)

    def init_parameters(self, train_data):
        """init parameters from training data"""
        # word 2 indices and tag 2 indices
        self.w2i = {}  # word to index
        self.c2i = {}  # char to index
        self.task2tag2idx = {}  # id of the task -> tag2idx

        self.w2i[UNK] = 0  # unk word / OOV
        self.c2i[UNK] = 0  # unk char
        self.c2i[WORD_START] = 1  # word start
        self.c2i[WORD_END] = 2  # word end index

        # word and char counters
        self.wcount = Counter()
        self.ccount = Counter()

        for seq in train_data:
            self.wcount.update([w for w in seq.words])
            self.ccount.update([c for w in seq.words for c in w])

            if seq.task_id not in self.task2tag2idx:
                self.task2tag2idx[seq.task_id] = {"<START>": START_TAG, "<END>": END_TAG}

            # record words and chars
            for word, tag in zip(seq.words, seq.tags):
                if word not in self.w2i:
                    self.w2i[word] = len(self.w2i)

                if self.c_in_dim > 0:
                    for char in word:
                        if char not in self.c2i:
                            self.c2i[char] = len(self.c2i)

                if tag not in self.task2tag2idx[seq.task_id]:
                    self.task2tag2idx[seq.task_id][tag] = len(self.task2tag2idx[seq.task_id])

        n = int(len(self.w2i) * 0.3) # top 30%
        print("Caching top {} words".format(n))
        for word in self.wcount.most_common(n):
            self.w2c_cache[word] = array.array('I', [self.c2i[WORD_START]]) + array.array('I', [self.get_c_idx(c) for c in word]) + array.array('I', [self.c2i[WORD_END]])
        # get total counts
        self.wtotal = np.sum([self.wcount[w] for w in self.wcount])
        self.ctotal = np.sum([self.ccount[c] for c in self.ccount])
        print("{} w features, {} c features".format(len(self.w2i), len(self.c2i)))
        #print(self.wtotal, self.ctotal)


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


    def save_lex_embeds(self, out_filename):
        """
        save final embeddings to file
        :param out_filename: filename
        """
        # construct reverse mapping
        i2l = {self.l2i[w]: w for w in self.l2i.keys()}

        OUT = open(out_filename+".l.emb","w")
        for lex_id in i2l.keys():
            lembeds_expression = self.lembeds[lex_id]
            lex = i2l[lex_id]
            OUT.write("{} {}\n".format(lex," ".join([str(x) for x in lembeds_expression.npvalue()])))
        OUT.close()


    def save_cw_embeds(self, out_filename):
        """
        save final character-based word-embeddings to file
        :param out_filename: filename
        """
        # construct reverse mapping using word embeddings
        i2cw = {self.w2i[w]: w for w in self.w2i.keys()}

        OUT = open(out_filename+".cw.emb","w")
        for word_id in i2cw.keys():
            word = i2cw[word_id]
            cwembeds = [v.npvalue()[0] for v in self.get_c_repr(word)]
            OUT.write("{} {}\n".format(word," ".join([str(x) for x in cwembeds])))
        OUT.close()


    def save_wordlex_map(self, out_filename):
        """
        save final word-to-lexicon-embedding map to file
        :param out_filename: filename
        """
        # construct reverse mapping using word embeddings
        i2wl = {self.w2i[w]: w for w in self.w2i.keys()}

        OUT = open(out_filename+".wlmap.emb","w")
        for word_id in i2wl.keys():
            word = i2wl[word_id]

            lex_feats = []
            for property in self.dictionary_values:
                values = is_in_dict(word, self.dictionary)
                if values:
                    if property in values:
                        lex_feats.append(property)
                    else:
                        lex_feats.append(UNK)
                else:
                    lex_feats.append(UNK) # unknown word

            OUT.write("{} {}\n".format(word," ".join([str(x) for x in lex_feats])))
        OUT.close()
        
    def save_transition_matrix(self, out_filename):
        """
        save transition matrix
        :param out_filename: filename
        """
        for task_id in self.predictors["output_layers_dict"].keys():
            output_predictor = self.predictors["output_layers_dict"][task_id]
            output_predictor.save_parameters(out_filename)



if __name__=="__main__":
    main()
