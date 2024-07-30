import os
import experiment

# track_id = 9001
# for i in range(2):
#     os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
#     P = type('Parameters', (object,), {})()
#     P.dataname = 'METRLA'
#     P.model = 'gwnet'
#     P.pre_model = 'TCN'
#     P.track_id = track_id
#     P.replication = i + 1
#     P.seed = 10

#     P.t_train = 0.4
#     P.t_val = 0.3
#     P.s_train = 0.7
#     P.s_val = 0.1
#     P.fold = 4

#     P.timestep_in = 12
#     P.timestep_out = 12
#     P.n_channel = 1
#     P.batch_size = 64

#     P.lstm_hidden_dim = 128
#     P.lstm_layers = 2
#     P.lstm_dropout = 0.2
#     P.gwnet_is_adp_adj = True
#     P.gwnet_is_SGA = False

#     P.adj_type = 'doubletransition'
#     P.adj_method = 1
#     P.adj_diag = 0
#     P.cost_kernals = [1, 2, 4, 8, 16]
#     P.cost_alpha = 0.5
#     P.cl_temperature = 1.4
#     P.is_pretrain = False
#     P.is_GCN_encoder = False
#     P.is_GCN_after_CL = False
#     P.gcn_order = 2
#     P.gcn_dropout = 0
#     P.augmentation = 'input_smoothing'
#     P.temporal_shifting_r = 0.8
#     P.input_smoothing_r = 0.9
#     P.input_smoothing_e = 20
#     P.encoder_to_model_ratio = 1
#     P.is_concat_encoder_model = False
#     P.is_layer_after_concat = False
#     P.is_always_augmentation = True

#     P.tolerance = 50
#     P.learn_rate = 0.001
#     P.pretrain_epoch = 50
#     P.train_epoch = 50
#     P.weight_decay = 0
#     P.is_testunseen = True
#     P.train_model_datasplit = 'B'
#     P.train_encoder_on = 'gpu'

#     P.is_mongo = False
#     P.example_verbose = False
#     P.is_tune = False
#     # Execute the experiment script
#     val_loss = experiment.main(P)

# track_id = 9002
# for i in range(5):
#     os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
#     P = type('Parameters', (object,), {})()
#     P.dataname = 'METRLA'
#     P.model = 'gwnet'
#     P.pre_model = 'TCN'
#     P.track_id = track_id
#     P.replication = i + 1
#     P.seed = 10

#     P.t_train = 0.4
#     P.t_val = 0.3
#     P.s_train = 0.7
#     P.s_val = 0.1
#     P.fold = 4

#     P.timestep_in = 12
#     P.timestep_out = 12
#     P.n_channel = 1
#     P.batch_size = 64

#     P.lstm_hidden_dim = 128
#     P.lstm_layers = 2
#     P.lstm_dropout = 0.2
#     P.gwnet_is_adp_adj = True
#     P.gwnet_is_SGA = False

#     P.adj_type = 'doubletransition'
#     P.adj_method = 1
#     P.adj_diag = 0
#     P.cost_kernals = [1, 2, 4, 8, 16]
#     P.cost_alpha = 0.5
#     P.cl_temperature = 2
#     P.is_pretrain = True
#     P.is_GCN_encoder = False
#     P.is_GCN_after_CL = False
#     P.gcn_order = 2
#     P.gcn_dropout = 0
#     P.augmentation = 'input_smoothing'
#     P.temporal_shifting_r = 0.9
#     P.input_smoothing_r = 0.7
#     P.input_smoothing_e = 120
#     P.encoder_to_model_ratio = 1.3
#     P.is_concat_encoder_model = True
#     P.is_layer_after_concat = True
#     P.is_always_augmentation = False

#     P.tolerance = 50
#     P.learn_rate = 0.001
#     P.pretrain_epoch = 50
#     P.train_epoch = 50
#     P.weight_decay = 0
#     P.is_testunseen = True
#     P.train_model_datasplit = 'B'
#     P.train_encoder_on = 'gpu'

#     P.is_mongo = False
#     P.example_verbose = False
#     P.is_tune = False
#     # Execute the experiment script
#     val_loss = experiment.main(P)

# track_id = 9004
# for i in range(4, 5):
#     os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
#     P = type('Parameters', (object,), {})()
#     P.dataname = 'METRLA'
#     P.model = 'gwnet'
#     P.pre_model = 'TCN'
#     P.track_id = track_id
#     P.replication = i + 1
#     P.seed = 10

#     P.t_train = 0.4
#     P.t_val = 0.3
#     P.s_train = 0.7
#     P.s_val = 0.1
#     P.fold = 4

#     P.timestep_in = 12
#     P.timestep_out = 12
#     P.n_channel = 1
#     P.batch_size = 64

#     P.lstm_hidden_dim = 128
#     P.lstm_layers = 2
#     P.lstm_dropout = 0.2
#     P.gwnet_is_adp_adj = True
#     P.gwnet_is_SGA = False

#     P.adj_type = 'doubletransition'
#     P.adj_method = 1
#     P.adj_diag = 0
#     P.cost_kernals = [1, 2, 4, 8, 16]
#     P.cost_alpha = 0.5
#     P.cl_temperature = 1.4
#     P.is_pretrain = True
#     P.is_GCN_encoder = False
#     P.is_GCN_after_CL = False
#     P.gcn_order = 2
#     P.gcn_dropout = 0
#     P.augmentation = 'input_smoothing'
#     P.temporal_shifting_r = 0.8
#     P.input_smoothing_r = 0.9
#     P.input_smoothing_e =500
#     P.encoder_to_model_ratio = 1
#     P.is_concat_encoder_model = True
#     P.is_layer_after_concat = False
#     P.is_always_augmentation = True

#     P.tolerance = 50
#     P.learn_rate = 0.001
#     P.pretrain_epoch = 50
#     P.train_epoch = 50
#     P.weight_decay = 0
#     P.is_testunseen = True
#     P.train_model_datasplit = 'B'
#     P.train_encoder_on = 'gpu'

#     P.is_mongo = False
#     P.example_verbose = False
#     P.is_tune = False
#     # Execute the experiment script
#     val_loss = experiment.main(P)

track_id = 9005
for i in range(5):
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    P = type('Parameters', (object,), {})()
    P.dataname = 'HAGUE'
    P.model = 'gwnet'
    P.pre_model = 'TCN'
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

    P.adj_type = 'doubletransition'
    P.adj_method = 1
    P.adj_diag = 0
    P.cost_kernals = [1, 2, 4, 8, 16]
    P.cost_alpha = 0.5
    P.cl_temperature = 1.4
    P.is_pretrain = False
    P.is_GCN_encoder = False
    P.is_GCN_after_CL = False
    P.gcn_order = 1
    P.gcn_dropout = 0
    P.augmentation = 'temporal_shifting'
    P.temporal_shifting_r = 0.8
    P.input_smoothing_r = 0.9
    P.input_smoothing_e = 20
    P.encoder_to_model_ratio = 1
    P.is_concat_encoder_model = False
    P.is_layer_after_concat = False
    P.is_always_augmentation = True

    P.tolerance = 50
    P.learn_rate = 0.001
    P.pretrain_epoch = 50
    P.train_epoch = 50
    P.weight_decay = 0
    P.is_testunseen = True
    P.train_model_datasplit = 'B'
    P.train_encoder_on = 'gpu'

    P.is_mongo = False
    P.example_verbose = False
    P.is_tune = False
    # Execute the experiment script
    val_loss = experiment.main(P)