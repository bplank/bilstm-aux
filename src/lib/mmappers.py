import _dynet as dynet
"""
various helper mappings
"""
## DyNet adds init option to choose initializer: https://github.com/clab/dynet/blob/master/python/CHANGES.md
INITIALIZER_MAP = {
                    'glorot': dynet.GlorotInitializer(),
                    'constant': dynet.ConstInitializer(0.01),
                    'uniform': dynet.UniformInitializer(0.1),
                    'normal': dynet.NormalInitializer(mean = 0, var = 1)
                  }

TRAINER_MAP = {
            "sgd": dynet.SimpleSGDTrainer,
            "adam": dynet.AdamTrainer,
            "adadelta": dynet.AdadeltaTrainer,
            "adagrad": dynet.AdagradTrainer,
            "momentum": dynet.MomentumSGDTrainer
           }

ACTIVATION_MAP = {
             "tanh": dynet.tanh,
             "rectify": dynet.rectify
           }

BUILDERS = {
            "lstm": dynet.LSTMBuilder, # is dynet.VanillaLSTMBuilder (cf. https://github.com/clab/dynet/issues/474)
            "lstmc": dynet.CoupledLSTMBuilder,
            "gru": dynet.GRUBuilder,
            "rnn": dynet.SimpleRNNBuilder
           }
