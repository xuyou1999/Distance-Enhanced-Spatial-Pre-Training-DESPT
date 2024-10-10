import os
import experiment

track_id = 9023 # set the start of the track id, each iteration will increase the track id by 1 so that each experiment has a unique track id
for i in range(5):
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    P = type('Parameters', (object,), {})()
    P.dataname = 'HAGUE_20_3' # dataset name
    P.model = 'gwnet' # model name, could be 'gwnet' or 'lstm', 'gwnet' is the default model in the paper
    P.pre_model = 'COST' # pretrain model framework options, could be 'COST' (STD) or 'TCN' (regular)
    P.track_id = track_id
    P.replication = i + 1
    P.seed = 10

    P.t_train = 0.4
    P.t_val = 0.3
    P.s_train = 0.7
    P.s_val = 0.1
    P.fold = 4

    P.timestep_in = 12
    P.timestep_out = 12
    P.n_channel = 1
    P.batch_size = 64

    P.lstm_hidden_dim = 128
    P.lstm_layers = 2
    P.lstm_dropout = 0.2
    P.gwnet_is_adp_adj = True
    P.gwnet_is_SGA = False

    P.adj_type = 'doubletransition' # set to best after tuning
    P.adj_method = 1 # set to best after tuning
    P.adj_diag = 0 # set to best after tuning
    P.cost_kernals = [1, 2, 4, 8, 16] # set to best after tuning
    P.cost_alpha = 2 # set to best after tuning
    P.cl_temperature = 1 # set to best after tuning
    P.is_pretrain = False # whether to pretrain the model, when True, the model will be 'SCPT+' or 'DESPT'
    P.is_GCN_encoder = False # whether to use GCN encoder, when True, the model will be 'DESPT'
    P.is_GCN_after_CL = False
    P.gcn_order = 1 # set to best after tuning
    P.gcn_dropout = 0 # set to best after tuning
    P.augmentation = 'temporal_shifting' # set to best after tuning
    P.temporal_shifting_r = 0.9 # set to best after tuning
    P.input_smoothing_r = 0.95 # set to best after tuning
    P.input_smoothing_e = 250 # set to best after tuning
    P.encoder_to_model_ratio = 2 # set to best after tuning
    P.is_concat_encoder_model = False # set to best after tuning
    P.is_layer_after_concat = False # set to best after tuning
    P.is_always_augmentation = False # set to best after tuning

    P.tolerance = 20 # equal to the number of epochs without improvement before early stopping, used for runtime study
    P.learn_rate = 0.001
    P.pretrain_epoch = 100
    P.train_epoch = 100
    P.weight_decay = 0
    P.is_testunseen = True
    P.train_model_datasplit = 'B'
    P.train_encoder_on = 'gpu'

    P.is_mongo = False # whether to use mongodb to store the experiment results
    P.example_verbose = False
    P.is_tune = False # Only True when tuning

    val_loss = experiment.main(P)