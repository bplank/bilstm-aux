"""
my NN library
(based on Yoav's)
"""
import _dynet as dynet
import numpy as np
import array
from lib.constants import START_TAG, END_TAG
from scipy import linalg

def init_dynet(seed):
    """initialize DyNet"""
    dyparams = dynet.DynetParams()
    # Fetch the command line arguments (optional)
    dyparams.from_args()
    # Set some parameters manualy (see the command line arguments documentation)
    dyparams.set_random_seed(seed)
    # Initialize with the given parameters
    dyparams.init()
    return dyparams


def pick_neg_log(pred, gold):
    return -dynet.log(dynet.pick(pred, gold))


def is_in_dict(word, dictionary):
    """ dictionary lookup """
    if word in dictionary:
        return dictionary[word]
    if word.lower() in dictionary:
        return dictionary[word.lower()]
    return False

## NN classes
class SequencePredictor:
    def __init__(self):
        pass
    
    def predict_sequence(self, inputs):
        raise NotImplementedError("SequencePredictor predict_sequence: Not Implemented")


class OutputSequencePredictor:
    def __init__(self):
        pass

    def predict_sequence(self, seq, inputs):
        raise NotImplementedError("SequencePredictor predict_sequence: Not Implemented")


class FFSequencePredictor(OutputSequencePredictor):
    """
    Local output predictor (softmax per tag)
    """
    def __init__(self, tag2index, network_builder):
        self.network_builder = network_builder
        self.tag2index = tag2index
        self.index2tag = {self.tag2index[t]: t for t in self.tag2index.keys()}


    def prune_softmax(self, softmax_distr, word, dictionary):
        ## implement type-constraint decoding
        if is_in_dict(word, dictionary):
            allowed_tag_indices = [self.tag2index[tag] for tag in is_in_dict(word, dictionary) if tag in self.tag2index]
            if len(allowed_tag_indices) > 1:
                for tag_idx in self.index2tag.keys():
                    if tag_idx not in allowed_tag_indices:
                        softmax_distr[tag_idx] = 0
#                        print(len([x for x in softmax_distr if x ==0]))
        return softmax_distr

    def predict_sequence(self, seq, inputs, train=False, output_confidences=False, unk_tag=None, dictionary=None, type_constraint=False, **kwargs):
        output = [self.network_builder(x, **kwargs) for x in inputs]
        if not train:
            if dictionary and type_constraint: # to type constraint decoding only during testing
                pred_tags = []
                for i, o in enumerate(output):
                    softmax_distr = o.npvalue()
                    word = seq.words[i]
                    softmax_distr = self.prune_softmax(softmax_distr, word, dictionary)
                    tag_best = self.index2tag[np.argmax(softmax_distr)]
                    pred_tags.append(tag_best)
                seq.pred_tags = pred_tags
            else:
                seq.pred_tags = [self.index2tag[np.argmax(o.npvalue())] for o in output]  # logprobs to indices
        if output_confidences:
            seq.tag_confidences = array.array('f', [np.max(o.npvalue()) for o in output])
        if train:
            # return loss per tag
            gold_tag_indices = array.array('I',[self.tag2index[t] for t in seq.tags])
            return dynet.esum([pick_neg_log(pred,gold) for pred, gold in zip(output, gold_tag_indices)])

    def save_parameters(self, out_filename):
        pass

class CRFSequencePredictor(OutputSequencePredictor):
    """
    Global output predictor
    """
    def __init__(self, model, num_tags, tag2index, network_builder, viterbi_loss=False):
        self.network_builder = network_builder # the per-class layers
        self.tag2index = tag2index
        self.index2tag = {self.tag2index[t]: t for t in self.tag2index.keys()}
        self.viterbi_loss=viterbi_loss

        self.num_tags = num_tags
        # Transition matrix for tagging layer, transitioning *to* i *from* j.
        self.trans_mat = model.add_lookup_parameters((num_tags, num_tags))  # tags x tags

    def save_parameters(self, out_filename):
        # save transition matrix
        OUT = open(out_filename + ".trans.mat", "w")
        for tag in self.index2tag.keys():
            for tag_prev in self.index2tag.keys():
                tag2tag_expression = self.trans_mat[tag_prev][tag]
                tag_prev_name = self.index2tag[tag_prev]
                tag_i_name = self.index2tag[tag]
                OUT.write("{} {} {}\n".format(tag_prev_name, tag_i_name, " ".join([str(x) for x in tag2tag_expression.npvalue()])))
        OUT.close()

        #np.savetxt(out_filename+'.matrix.out', self.trans_mat.npvalue(), delimiter=',')
        print("done.")

    def predict_sequence(self, seq, inputs, train=False, output_confidences=False, unk_tag=None, dictionary=None, type_constraint=False, **kwargs):
        score_vecs = [self.network_builder(x, **kwargs) for x in inputs]

        if not train:
            #pred_tag_indices = self.viterbi(start_b, T, end_b, score_vecs)
            pred_tag_indices, tag_scores = self.viterbi(score_vecs, unk_tag=unk_tag, dictionary=dictionary)
            seq.pred_tags = [self.index2tag[t] for t in pred_tag_indices]
            if output_confidences:
                print("not implemented")
            return
        else:
            if self.viterbi_loss:
                pred_tag_indices, path_score = self.viterbi(score_vecs)
                instance_score = path_score #viterbi score
            else:
                forward_score = self.forward(score_vecs)
                instance_score = forward_score
            # return loss
            gold_tag_indices = array.array('I',[self.tag2index[t] for t in seq.tags])
            # decode CRF
            gold_score = self.score_sentence(score_vecs, gold_tag_indices)
            return instance_score - gold_score
            # return normalizer - gold_score

    # code adapted from K.Stratos' code basis
    def score_sentence(self, score_vecs, tags):
        assert(len(score_vecs)==len(tags))
        tags.insert(0, START_TAG) # add start
        total = dynet.scalarInput(.0)
        for i, obs in enumerate(score_vecs):
            # transition to next from i and emission
            next_tag = tags[i + 1]
            total += dynet.pick(self.trans_mat[next_tag],tags[i]) + dynet.pick(obs,next_tag)
        total += dynet.pick(self.trans_mat[END_TAG],tags[-1])
        return total

    # code based on https://github.com/rguthrie3/BiLSTM-CRF
    def viterbi(self, observations, unk_tag=None, dictionary=None):
        #if dictionary:
        #    raise NotImplementedError("type constraints not yet implemented for CRF")
        backpointers = []
        init_vvars   = [-1e10] * self.num_tags
        init_vvars[START_TAG] = 0 # <Start> has all the probability
        for_expr     = dynet.inputVector(init_vvars)
        trans_exprs  = [self.trans_mat[idx] for idx in range(self.num_tags)]
        for obs in observations:
            bptrs_t = []
            vvars_t = []
            for next_tag in range(self.num_tags):
                next_tag_expr = for_expr + trans_exprs[next_tag]
                next_tag_arr = next_tag_expr.npvalue()
                best_tag_id  = np.argmax(next_tag_arr)
                if unk_tag:
                    best_tag = self.index2tag[best_tag_id]
                    if best_tag == unk_tag:
                        next_tag_arr[np.argmax(next_tag_arr)] = 0 # set to 0
                        best_tag_id = np.argmax(next_tag_arr) # get second best

                bptrs_t.append(best_tag_id)
                vvars_t.append(dynet.pick(next_tag_expr, best_tag_id))
            for_expr = dynet.concatenate(vvars_t) + obs
            backpointers.append(bptrs_t)
        # Perform final transition to terminal
        terminal_expr = for_expr + trans_exprs[END_TAG]
        terminal_arr  = terminal_expr.npvalue()
        best_tag_id   = np.argmax(terminal_arr)
        path_score    = dynet.pick(terminal_expr, best_tag_id)
        # Reverse over the backpointers to get the best path
        best_path = [best_tag_id] # Start with the tag that was best for terminal
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        start = best_path.pop() # Remove the start symbol
        best_path.reverse()
        assert start == START_TAG
        # Return best path and best path's score
        return best_path, path_score

    def forward(self, observations):
        # calculate forward pass
        def log_sum_exp(scores):
            npval = scores.npvalue()
            argmax_score = np.argmax(npval)
            max_score_expr = dynet.pick(scores, argmax_score)
            max_score_expr_broadcast = dynet.concatenate([max_score_expr] * self.num_tags)
            return max_score_expr + dynet.logsumexp_dim((scores - max_score_expr_broadcast),0)

        init_alphas = [-1e10] * self.num_tags
        init_alphas[START_TAG] = 0
        for_expr = dynet.inputVector(init_alphas)
        for obs in observations:
            alphas_t = []
            for next_tag in range(self.num_tags):
                obs_broadcast = dynet.concatenate([dynet.pick(obs, next_tag)] * self.num_tags)
                next_tag_expr = for_expr + self.trans_mat[next_tag] + obs_broadcast
                alphas_t.append(log_sum_exp(next_tag_expr))
            for_expr = dynet.concatenate(alphas_t)
        terminal_expr = for_expr + self.trans_mat[END_TAG]
        alpha = log_sum_exp(terminal_expr)
        return alpha


class RNNSequencePredictor(SequencePredictor):
    def __init__(self, rnn_builder):
        """
        rnn_builder: a LSTMBuilder/SimpleRNNBuilder or GRU builder object
        """
        self.builder = rnn_builder
        
    def predict_sequence(self, inputs):
        s_init = self.builder.initial_state()
        return s_init.transduce(inputs)


class BiRNNSequencePredictor(SequencePredictor):
    """ a bidirectional RNN (LSTM/GRU) """
    def __init__(self, f_builder, b_builder):
        self.f_builder = f_builder
        self.b_builder = b_builder

    def predict_sequence(self, f_inputs, b_inputs):
        f_init = self.f_builder.initial_state()
        b_init = self.b_builder.initial_state()
        forward_sequence = f_init.transduce(f_inputs)
        backward_sequence = b_init.transduce(reversed(b_inputs))
        return forward_sequence, backward_sequence 


class Layer:
    """ Class for affine layer transformation or two-layer MLP """
    def __init__(self, model, in_dim, output_dim, activation=dynet.tanh, mlp=0, mlp_activation=dynet.rectify):
        # if mlp > 0, add a hidden layer of that dimension
        self.act = activation
        self.mlp = mlp
        if mlp:
            print('>>> use mlp with dim {} ({})<<<'.format(mlp, mlp_activation))
            mlp_dim = mlp
            self.mlp_activation = mlp_activation
            self.W_mlp = model.add_parameters((mlp_dim, in_dim))
            self.b_mlp = model.add_parameters((mlp_dim))
        else:
            mlp_dim = in_dim
        self.W = model.add_parameters((output_dim, mlp_dim))
        self.b = model.add_parameters((output_dim))
        
    def __call__(self, x, soft_labels=False, temperature=None, train=False):
        if self.mlp:
            W_mlp = dynet.parameter(self.W_mlp)
            b_mlp = dynet.parameter(self.b_mlp)
            act = self.mlp_activation
            x_in = act(W_mlp * x + b_mlp)
        else:
            x_in = x
        # from params to expressions
        W = dynet.parameter(self.W)
        b = dynet.parameter(self.b)

        logits = W*x_in + b
        if soft_labels and temperature:
            # calculate the soft labels smoothed with the temperature
            # see Distilling the Knowledge in a Neural Network
            elems = dynet.exp(logits / temperature)
            return dynet.cdiv(elems, dynet.sum_elems(elems))
        if self.act:
            return self.act(logits)
        return logits




