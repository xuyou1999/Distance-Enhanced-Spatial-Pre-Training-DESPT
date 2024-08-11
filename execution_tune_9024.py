import os
import experiment

temporal_shifting_r = [0.7, 0.8, 0.9, 0.95]
input_smoothing_r = [0.7, 0.8, 0.9, 0.95]
input_smoothing_e = [10, 20, 40, 80, 150, 250, 400]
augmentation = ['input_smoothing', 'temporal_shifting']
cost_kernals = [[1, 2, 4], [1, 2, 4, 8], [1, 2, 4, 8, 16]]
cost_alpha = [0.3, 0.5, 0.7, 0.9, 1, 2]
cl_temperature = [0.05, 0.1, 0.5, 0.8, 1, 2]
encoder_to_model_ratio = [0.4, 0.6, 0.8, 1, 1.2, 1.5, 2]
gcn_order = [1, 2]
gcn_dropout = [0, 0.1, 0.2, 0.4]
adj_method = [1, 2]
adj_diag = [0, 1]
is_concat_encoder_model = [True, False]
is_layer_after_concat = [True, False]
is_always_augmentation = [True, False]

track_id = 1640

best_temporal_shifting_r = 0.7
best_input_smoothing_r = 0.9
best_input_smoothing_e = 250
best_augmentation = 'temporal_shifting'

best_cost_kernals = [1, 2, 4, 8]
best_cost_alpha = 0.5

best_cl_temperature = 2
best_encoder_to_model_ratio = 1.5

best_gcn_order = 1
best_gcn_dropout = 0
best_adj_method = 1
best_adj_diag = 0

best_is_concat_encoder_model = False
best_is_layer_after_concat = False


# Always augmentation
best_is_always_augmentation = None
min_loss = float('inf')
for value in is_always_augmentation:
    losses = []
    for i in range(1):
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
        P = type('Parameters', (object,), {})()
        P.dataname = 'HAGUE_20_3'
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
        P.adj_method = best_adj_method
        P.adj_diag = best_adj_diag
        P.cost_kernals = best_cost_kernals
        P.cost_alpha = best_cost_alpha
        P.cl_temperature = best_cl_temperature
        P.is_pretrain = True
        P.is_GCN_encoder = False
        P.is_GCN_after_CL = False
        P.gcn_order = best_gcn_order
        P.gcn_dropout = best_gcn_dropout
        P.augmentation = best_augmentation
        P.temporal_shifting_r = best_temporal_shifting_r
        P.input_smoothing_r = best_input_smoothing_r
        P.input_smoothing_e = best_input_smoothing_e
        P.encoder_to_model_ratio = best_encoder_to_model_ratio
        P.is_concat_encoder_model = best_is_concat_encoder_model
        P.is_layer_after_concat = best_is_layer_after_concat
        P.is_always_augmentation = value

        P.tolerance = 20
        P.learn_rate = 0.001
        P.pretrain_epoch = 100
        P.train_epoch = 100
        P.weight_decay = 0
        P.is_testunseen = True
        P.train_model_datasplit = 'B'
        P.train_encoder_on = 'gpu'

        P.is_mongo = False
        P.example_verbose = False
        P.is_tune = True
        
        # Execute the experiment script
        val_loss = experiment.main(P)
        losses.append(val_loss)
    final_loss = sum(losses) / len(losses)
    if final_loss < min_loss:
        best_is_always_augmentation = value
        min_loss = final_loss
    track_id += 1
print('Best parameter is_always_augmentation:', best_is_always_augmentation)

# Test
for i in range(5):
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    P = type('Parameters', (object,), {})()
    P.dataname = 'HAGUE_20_3'
    P.model = 'gwnet'
    P.pre_model = 'TCN'
    P.track_id = 9024
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
    P.adj_method = best_adj_method
    P.adj_diag = best_adj_diag
    P.cost_kernals = best_cost_kernals
    P.cost_alpha = best_cost_alpha
    P.cl_temperature = best_cl_temperature
    P.is_pretrain = True
    P.is_GCN_encoder = False
    P.is_GCN_after_CL = False
    P.gcn_order = best_gcn_order
    P.gcn_dropout = best_gcn_dropout
    P.augmentation = best_augmentation
    P.temporal_shifting_r = best_temporal_shifting_r
    P.input_smoothing_r = best_input_smoothing_r
    P.input_smoothing_e = best_input_smoothing_e
    P.encoder_to_model_ratio = best_encoder_to_model_ratio
    P.is_concat_encoder_model = best_is_concat_encoder_model
    P.is_layer_after_concat = best_is_layer_after_concat
    P.is_always_augmentation = best_is_always_augmentation

    P.tolerance = 20
    P.learn_rate = 0.001
    P.pretrain_epoch = 100
    P.train_epoch = 100
    P.weight_decay = 0
    P.is_testunseen = True
    P.train_model_datasplit = 'B'
    P.train_encoder_on = 'gpu'

    P.is_mongo = False
    P.example_verbose = False
    P.is_tune = False
    
    # Execute the experiment script
    val_loss = experiment.main(P)