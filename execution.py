import os
import experiment

for i in range(1):
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    P = type('Parameters', (object,), {})()
    P.dataname = 'METRLA'
    P.model = 'LSTM'
    P.pre_model = 'TCN'
    P.track_id = 0
    P.replication = i + 1
    P.seed = 10

    P.t_train = 0.2
    P.t_val = 0.15
    P.s_train = 0.7
    P.s_val = 0.1
    P.fold = 2

    P.timestep_in = 12
    P.timestep_out = 12
    P.n_channel = 1
    P.batch_size = 64

    P.lstm_hidden_dim = 128
    P.lstm_layers = 2
    P.lstm_dropout = 0.2
    P.gwnet_is_adp_adj = True
    P.gwnet_is_SGA = False

    P.adj_type = 'doubletransition'
    P.adj_method = 1
    P.adj_diag = 0
    P.cost_kernals = [1, 2, 4, 8, 16]
    P.cost_alpha = 0.5
    P.cl_temperature = 1.4
    P.is_pretrain = True
    P.is_GCN_encoder = False
    P.is_GCN_after_CL = False
    P.gcn_order = 2
    P.gcn_dropout = 0
    P.augmentation = 'input_smoothing'
    P.temporal_shifting_r = 0.8
    P.input_smoothing_r = 0.9
    P.input_smoothing_e = 20
    P.encoder_to_model_ratio = 1
    P.is_concat_encoder_model = True
    P.is_layer_after_concat = False
    P.is_always_augmentation = True

    P.tolerance = 10
    P.learn_rate = 0.001
    P.pretrain_epoch = 2
    P.train_epoch = 2
    P.weight_decay = 0
    P.is_testunseen = True
    P.train_model_datasplit = 'B'
    P.train_encoder_on = 'gpu'

    P.is_mongo = True
    P.example_verbose = False
    P.is_tune = True

    # Execute the experiment script
    experiment.main(P)