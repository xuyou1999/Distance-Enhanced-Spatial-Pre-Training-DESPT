import os
import experiment

adj_method = [1, 2]
adj_diag = [0, 1]
cl_temperature = [0.8, 1, 1.1, 1,2, 1,3, 1,5, 1,7, 2, 3]
gcn_order = [1, 2]
gcn_dropout = [0, 0.1, 0.2, 0.4]
augmentation = ['input_smoothing', 'temporal_shifting']
temporal_shifting_r = [0.6, 0.8, 1]
input_smoothing_r = [0.8, 0.9, 1]
input_smoothing_e = [10, 20, 30]
encoder_to_model_ratio = [0.4, 0.6, 0.8, 1, 1.2, 1.5, 2]
is_concat_encoder_model = [True, False]
is_always_augmentation = [True, False]

track_id = 200

best_adj_method = None
min_loss = float('inf')

for adj_method_value in adj_method:
    losses = []
    for i in range(3):
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
        P = type('Parameters', (object,), {})()
        P.dataname = 'HAGUE'
        P.model = 'LSTM'
        P.pre_model = 'TCN'
        P.track_id = track_id
        P.replication = i + 1
        P.seed = 10

        P.t_train = 0.4
        P.t_val = 0.3
        P.s_train = 0.7
        P.s_val = 0.1
        P.fold = 1

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
        best_adj_method = adj_method_value
        min_loss = final_loss
    track_id += 1
print('Best parameter:', best_adj_method)