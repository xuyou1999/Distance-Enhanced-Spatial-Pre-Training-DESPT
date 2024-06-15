import os
import experiment

# Tuning Pretrain wihtout GCN, for HAGUE dataset on GWNET model
temporal_shifting_r = [0.6, 0.8, 1]
input_smoothing_r = [0.8, 0.9, 1]
input_smoothing_e = [10, 20, 30]
augmentation = ['input_smoothing', 'temporal_shifting']
cl_temperature = [0.8, 1, 1.1, 1,2, 1,3, 1,5, 1,7, 2, 3]
encoder_to_model_ratio = [0.4, 0.6, 0.8, 1, 1.2, 1.5, 2]
gcn_order = [1, 2]
gcn_dropout = [0, 0.1, 0.2, 0.4]
adj_method = [1, 2]
adj_diag = [0, 1]
is_concat_encoder_model = [True, False]
is_always_augmentation = [True, False]

track_id = 200

# Temporal shifting ratio
best_temporal_shifting_r = None
min_loss = float('inf')
for value in temporal_shifting_r:
    losses = []
    for i in range(3):
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
        P.fold = 1

        P.timestep_in = 12
        P.timestep_out = 12
        P.n_channel = 1
        P.batch_size = 64

        P.lstm_hidden_dim = 128
        P.lstm_layers = 2
        P.lstm_dropout = 0.2
        P.gwnet_is_adp_adj = False
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
        P.augmentation = 'temporal_shifting'
        P.temporal_shifting_r = value
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
        best_temporal_shifting_r = value
        min_loss = final_loss
    track_id += 1
print('Best parameter temporal_shifting_r:', best_temporal_shifting_r)

# input smoothing ratio
best_input_smoothing_r = None
min_loss = float('inf')
for value in input_smoothing_r:
    losses = []
    for i in range(3):
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
        P.fold = 1

        P.timestep_in = 12
        P.timestep_out = 12
        P.n_channel = 1
        P.batch_size = 64

        P.lstm_hidden_dim = 128
        P.lstm_layers = 2
        P.lstm_dropout = 0.2
        P.gwnet_is_adp_adj = False
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
        P.temporal_shifting_r = best_temporal_shifting_r
        P.input_smoothing_r = value
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
        best_input_smoothing_r = value
        min_loss = final_loss
    track_id += 1
print('Best parameter input_smoothing_r:', best_input_smoothing_r)

# input smoothing epoch
best_input_smoothing_e = None
min_loss = float('inf')
for value in input_smoothing_e:
    losses = []
    for i in range(3):
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
        P.fold = 1

        P.timestep_in = 12
        P.timestep_out = 12
        P.n_channel = 1
        P.batch_size = 64

        P.lstm_hidden_dim = 128
        P.lstm_layers = 2
        P.lstm_dropout = 0.2
        P.gwnet_is_adp_adj = False
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
        P.temporal_shifting_r = best_temporal_shifting_r
        P.input_smoothing_r = best_input_smoothing_r
        P.input_smoothing_e = value
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
        best_input_smoothing_e = value
        min_loss = final_loss
    track_id += 1
print('Best parameter input_smoothing_e:', best_input_smoothing_e)

# Augmentation
best_augmentation = None
min_loss = float('inf')
for value in augmentation:
    losses = []
    for i in range(3):
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
        P.fold = 1

        P.timestep_in = 12
        P.timestep_out = 12
        P.n_channel = 1
        P.batch_size = 64

        P.lstm_hidden_dim = 128
        P.lstm_layers = 2
        P.lstm_dropout = 0.2
        P.gwnet_is_adp_adj = False
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
        P.augmentation = value
        P.temporal_shifting_r = best_temporal_shifting_r
        P.input_smoothing_r = best_input_smoothing_r
        P.input_smoothing_e = best_input_smoothing_e
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
        best_augmentation = value
        min_loss = final_loss
    track_id += 1
print('Best parameter augmentation:', best_augmentation)

# CL temperature
best_cl_temperature = None
min_loss = float('inf')
for value in cl_temperature:
    losses = []
    for i in range(3):
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
        P.fold = 1

        P.timestep_in = 12
        P.timestep_out = 12
        P.n_channel = 1
        P.batch_size = 64

        P.lstm_hidden_dim = 128
        P.lstm_layers = 2
        P.lstm_dropout = 0.2
        P.gwnet_is_adp_adj = False
        P.gwnet_is_SGA = False

        P.adj_type = 'doubletransition'
        P.adj_method = 1
        P.adj_diag = 0
        P.cost_kernals = [1, 2, 4, 8, 16]
        P.cost_alpha = 0.5
        P.cl_temperature = value
        P.is_pretrain = True
        P.is_GCN_encoder = False
        P.is_GCN_after_CL = False
        P.gcn_order = 2
        P.gcn_dropout = 0
        P.augmentation = best_augmentation
        P.temporal_shifting_r = best_temporal_shifting_r
        P.input_smoothing_r = best_input_smoothing_r
        P.input_smoothing_e = best_input_smoothing_e
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
        best_cl_temperature = value
        min_loss = final_loss
    track_id += 1
print('Best parameter cl_temperature:', best_cl_temperature)

# Encoder to model ratio
best_encoder_to_model_ratio = None
min_loss = float('inf')
for value in encoder_to_model_ratio:
    losses = []
    for i in range(3):
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
        P.fold = 1

        P.timestep_in = 12
        P.timestep_out = 12
        P.n_channel = 1
        P.batch_size = 64

        P.lstm_hidden_dim = 128
        P.lstm_layers = 2
        P.lstm_dropout = 0.2
        P.gwnet_is_adp_adj = False
        P.gwnet_is_SGA = False

        P.adj_type = 'doubletransition'
        P.adj_method = 1
        P.adj_diag = 0
        P.cost_kernals = [1, 2, 4, 8, 16]
        P.cost_alpha = 0.5
        P.cl_temperature = best_cl_temperature
        P.is_pretrain = True
        P.is_GCN_encoder = False
        P.is_GCN_after_CL = False
        P.gcn_order = 2
        P.gcn_dropout = 0
        P.augmentation = best_augmentation
        P.temporal_shifting_r = best_temporal_shifting_r
        P.input_smoothing_r = best_input_smoothing_r
        P.input_smoothing_e = best_input_smoothing_e
        P.encoder_to_model_ratio = value
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
        best_encoder_to_model_ratio = value
        min_loss = final_loss
    track_id += 1
print('Best parameter encoder_to_model_ratio:', best_encoder_to_model_ratio)

# GCN order
best_gcn_order = None
min_loss = float('inf')
for value in gcn_order:
    losses = []
    for i in range(3):
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
        P.fold = 1

        P.timestep_in = 12
        P.timestep_out = 12
        P.n_channel = 1
        P.batch_size = 64

        P.lstm_hidden_dim = 128
        P.lstm_layers = 2
        P.lstm_dropout = 0.2
        P.gwnet_is_adp_adj = False
        P.gwnet_is_SGA = False

        P.adj_type = 'doubletransition'
        P.adj_method = 1
        P.adj_diag = 0
        P.cost_kernals = [1, 2, 4, 8, 16]
        P.cost_alpha = 0.5
        P.cl_temperature = best_cl_temperature
        P.is_pretrain = True
        P.is_GCN_encoder = False
        P.is_GCN_after_CL = False
        P.gcn_order = value
        P.gcn_dropout = 0
        P.augmentation = best_augmentation
        P.temporal_shifting_r = best_temporal_shifting_r
        P.input_smoothing_r = best_input_smoothing_r
        P.input_smoothing_e = best_input_smoothing_e
        P.encoder_to_model_ratio = best_encoder_to_model_ratio
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
        best_gcn_order = value
        min_loss = final_loss
    track_id += 1
print('Best parameter gcn_order:', best_gcn_order)

# GCN dropout
best_gcn_dropout = None
min_loss = float('inf')
for value in gcn_dropout:
    losses = []
    for i in range(3):
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
        P.fold = 1

        P.timestep_in = 12
        P.timestep_out = 12
        P.n_channel = 1
        P.batch_size = 64

        P.lstm_hidden_dim = 128
        P.lstm_layers = 2
        P.lstm_dropout = 0.2
        P.gwnet_is_adp_adj = False
        P.gwnet_is_SGA = False

        P.adj_type = 'doubletransition'
        P.adj_method = 1
        P.adj_diag = 0
        P.cost_kernals = [1, 2, 4, 8, 16]
        P.cost_alpha = 0.5
        P.cl_temperature = best_cl_temperature
        P.is_pretrain = True
        P.is_GCN_encoder = False
        P.is_GCN_after_CL = False
        P.gcn_order = best_gcn_order
        P.gcn_dropout = value
        P.augmentation = best_augmentation
        P.temporal_shifting_r = best_temporal_shifting_r
        P.input_smoothing_r = best_input_smoothing_r
        P.input_smoothing_e = best_input_smoothing_e
        P.encoder_to_model_ratio = best_encoder_to_model_ratio
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
        best_gcn_dropout = value
        min_loss = final_loss
    track_id += 1
print('Best parameter gcn_dropout:', best_gcn_dropout)

# Adj method
best_adj_method = None
min_loss = float('inf')
for value in adj_method:
    losses = []
    for i in range(3):
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
        P.fold = 1

        P.timestep_in = 12
        P.timestep_out = 12
        P.n_channel = 1
        P.batch_size = 64

        P.lstm_hidden_dim = 128
        P.lstm_layers = 2
        P.lstm_dropout = 0.2
        P.gwnet_is_adp_adj = False
        P.gwnet_is_SGA = False

        P.adj_type = 'doubletransition'
        P.adj_method = value
        P.adj_diag = 0
        P.cost_kernals = [1, 2, 4, 8, 16]
        P.cost_alpha = 0.5
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
        best_adj_method = value
        min_loss = final_loss
    track_id += 1
print('Best parameter adj_method:', best_adj_method)

# Adj diag
best_adj_diag = None
min_loss = float('inf')
for value in adj_diag:
    losses = []
    for i in range(3):
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
        P.fold = 1

        P.timestep_in = 12
        P.timestep_out = 12
        P.n_channel = 1
        P.batch_size = 64

        P.lstm_hidden_dim = 128
        P.lstm_layers = 2
        P.lstm_dropout = 0.2
        P.gwnet_is_adp_adj = False
        P.gwnet_is_SGA = False

        P.adj_type = 'doubletransition'
        P.adj_method = best_adj_method
        P.adj_diag = value
        P.cost_kernals = [1, 2, 4, 8, 16]
        P.cost_alpha = 0.5
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
        best_adj_diag = value
        min_loss = final_loss
    track_id += 1
print('Best parameter adj_diag:', best_adj_diag)

# Concat encoder model
best_is_concat_encoder_model = None
min_loss = float('inf')
for value in is_concat_encoder_model:
    losses = []
    for i in range(3):
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
        P.fold = 1

        P.timestep_in = 12
        P.timestep_out = 12
        P.n_channel = 1
        P.batch_size = 64

        P.lstm_hidden_dim = 128
        P.lstm_layers = 2
        P.lstm_dropout = 0.2
        P.gwnet_is_adp_adj = False
        P.gwnet_is_SGA = False

        P.adj_type = 'doubletransition'
        P.adj_method = best_adj_method
        P.adj_diag = best_adj_diag
        P.cost_kernals = [1, 2, 4, 8, 16]
        P.cost_alpha = 0.5
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
        P.is_concat_encoder_model = value
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
        best_is_concat_encoder_model = value
        min_loss = final_loss
    track_id += 1
print('Best parameter is_concat_encoder_model:', best_is_concat_encoder_model)

# Always augmentation
best_is_always_augmentation = None
min_loss = float('inf')
for value in is_always_augmentation:
    losses = []
    for i in range(3):
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
        P.fold = 1

        P.timestep_in = 12
        P.timestep_out = 12
        P.n_channel = 1
        P.batch_size = 64

        P.lstm_hidden_dim = 128
        P.lstm_layers = 2
        P.lstm_dropout = 0.2
        P.gwnet_is_adp_adj = False
        P.gwnet_is_SGA = False

        P.adj_type = 'doubletransition'
        P.adj_method = best_adj_method
        P.adj_diag = best_adj_diag
        P.cost_kernals = [1, 2, 4, 8, 16]
        P.cost_alpha = 0.5
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
        P.is_layer_after_concat = False
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
