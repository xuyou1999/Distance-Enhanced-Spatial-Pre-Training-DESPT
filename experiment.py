import torch
import numpy as np
import pandas as pd
import os
import time
from datetime import datetime
import unseen_nodes
from utils import *
from encoder import *
from augmentation import *
from model import *
import Metrics

class StandardScaler:
    def __init__(self):
        self.u = None
        self.z = None
    def fit_transform(self, x):
        self.u = x.mean()
        self.z = x.std()
        return (x-self.u)/self.z
    def inverse_transform(self, x):
        return x * self.z + self.u

def save_parameters(param_obj, filename):
    # Create a dictionary from the parameter object attributes
    data = {attr: [getattr(param_obj, attr)] for attr in dir(param_obj) if not attr.startswith("__") and not callable(getattr(param_obj, attr))}
    # Ensure 'exe_id' is the first column if it exists
    column_order = ['exe_id'] + [col for col in data if col != 'exe_id']
    # Create a DataFrame from the new data, specifying the column order
    new_row = pd.DataFrame(data, columns=column_order)
    # Attempt to read the existing CSV file into a DataFrame. If it doesn't exist, create an empty DataFrame.
    try:
        df = pd.read_csv(filename)
    except FileNotFoundError:
        df = pd.DataFrame(columns=column_order)  # Ensure 'exe_id' is the first column in an empty DataFrame
    # Append the new data as a row to the DataFrame
    df = pd.concat([df, new_row], ignore_index=True)
    # Reorder columns before saving, to ensure 'exe_id' is the first column, especially if the CSV initially did not exist
    df = df[column_order]
    # Save the updated DataFrame back to the CSV, ensuring all columns are included
    df.to_csv(filename, index=False)

def getModel(name, device, support_len):
    if name == 'gwnet':
        model = gwnet(device, num_nodes=P.n_sensor, in_dim=P.n_channel, adp_adj=P.gwnet_is_adp_adj, sga=P.gwnet_is_SGA).to(device)
    elif name == 'LSTM':
        if P.is_pretrain == False:
            lstm_input_dim = 32
        elif P.is_concat_encoder_model:
            lstm_input_dim = 64
        else:
            lstm_input_dim = 32
        model = LSTM_uni(input_dim=P.n_channel, lstm_input_dim=lstm_input_dim, hidden_dim=P.lstm_hidden_dim, device=device, is_GCN_after_CL = P.is_GCN_after_CL, support_len = support_len).to(device)
    return model

def getXSYS(data, mode):
    TRAIN_NUM = int(data.shape[0] * (P.t_train + P.t_val))
    XS, YS = [], []
    if mode == 'TRAIN':    
        for i in range(TRAIN_NUM - P.timestep_out - P.timestep_in + 1):
            x = data[i:i+P.timestep_in, :]
            y = data[i+P.timestep_in:i+P.timestep_in+P.timestep_out, :]
            XS.append(x), YS.append(y)
    elif mode == 'TEST':
        for i in range(TRAIN_NUM - P.timestep_in,  data.shape[0] - P.timestep_out - P.timestep_in + 1):
            x = data[i:i+P.timestep_in, :]
            y = data[i+P.timestep_in:i+P.timestep_in+P.timestep_out, :]
            XS.append(x), YS.append(y)
    XS, YS = np.array(XS), np.array(YS)
    XS, YS = XS[:, :, :, np.newaxis], YS[:, :, :, np.newaxis]
    XS = XS.transpose(0, 3, 2, 1)
    return XS, YS

def setups(device):
    '''
    If the save folder does not exist, create it.
    '''
    if not os.path.exists(P.save_path):
        os.makedirs(P.save_path)
    '''
    Set the seed to control the randomness of the experiment.
    '''
    torch.manual_seed(P.seed)
    torch.cuda.manual_seed(P.seed)
    
    '''
    Split the data in temporal dimension into training and testing. 
    At this step, training set contains both training and validation data.
    So that each each dimension correspond to [instance, input, sensor, output]
    '''
    trainXS, trainYS = getXSYS(data, 'TRAIN')
    testXS, testYS = getXSYS(data, 'TEST')

    print('\ntrainXS.shape', trainXS.shape)
    print('trainYS.shape', trainYS.shape)
    print('testXS.shape', testXS.shape)
    print('testYS.shape', testYS.shape)

    if P.example_verbose:
        print('\nFirst instance trainXS for first sensor:', trainXS[0,0,0,:])
        print('\nFirst instance trainYS for first sensor:', trainYS[0,:,0,0])
        print('\nLast instance trainXS for first sensor:', trainXS[-1,0,0,:])
        print('\nFirst instance testXS for first sensor:', testXS[0,0,0,:])
        print('\nFirst instance trainXS for second sensor:', trainXS[0,0,1,:])

    '''
    Split the training set further into training and validation sets.
    '''
    P.trainval_size = len(trainXS)
    P.train_size = int(P.trainval_size * (P.t_train / (P.t_train + P.t_val)))
    XS_torch_trn = trainXS[:P.train_size,:,:,:]
    YS_torch_trn = trainYS[:P.train_size,:,:,:]
    XS_torch_val = trainXS[P.train_size:P.trainval_size,:,:,:]
    YS_torch_val = trainYS[P.train_size:P.trainval_size,:,:,:]

    '''
    Get the sensor indexes for each spatial splited set.
    '''
    spatialSplit_unseen = unseen_nodes.SpatialSplit(data.shape[1], r_trn=P.s_train, r_val=P.s_val, r_tst=(1-P.s_train-P.s_val), seed=P.seed)
    spatialSplit_allNod = unseen_nodes.SpatialSplit(data.shape[1], r_trn=P.s_train, r_val=min(1.0,P.s_val+P.s_train), r_tst=1.0, seed=P.seed)
    print('\nspatialSplit_unseen', spatialSplit_unseen)
    print('spatialSplit_allNod', spatialSplit_allNod)
    if P.example_verbose:
        print('\nspatialSplit_unseen.i_trn', spatialSplit_unseen.i_trn)
        print('spatialSplit_unseen.i_val', spatialSplit_unseen.i_val)
        print('spatialSplit_unseen.i_tst', spatialSplit_unseen.i_tst)
        print('spatialSplit_allNod.i_trn', spatialSplit_allNod.i_trn)
        print('spatialSplit_allNod.i_val', spatialSplit_allNod.i_val)
        print('spatialSplit_allNod.i_tst', spatialSplit_allNod.i_tst)
    XS_torch_train = torch.Tensor(XS_torch_trn[:,:,spatialSplit_allNod.i_trn,:])
    YS_torch_train = torch.Tensor(YS_torch_trn[:,:,spatialSplit_allNod.i_trn,:])
    XS_torch_train_model = torch.Tensor(XS_torch_val[:,:,spatialSplit_allNod.i_trn,:])
    YS_torch_train_model = torch.Tensor(YS_torch_val[:,:,spatialSplit_allNod.i_trn,:])
    XS_torch_val_u = torch.Tensor(XS_torch_val[:,:,spatialSplit_unseen.i_val,:])
    YS_torch_val_u = torch.Tensor(YS_torch_val[:,:,spatialSplit_unseen.i_val,:])
    XS_torch_val_a = torch.Tensor(XS_torch_val[:,:,spatialSplit_allNod.i_val,:])
    YS_torch_val_a = torch.Tensor(YS_torch_val[:,:,spatialSplit_allNod.i_val,:])
    XS_torch_tst_u = torch.Tensor(testXS[:,:,spatialSplit_unseen.i_tst,:])
    YS_torch_tst_u = torch.Tensor(testYS[:,:,spatialSplit_unseen.i_tst,:])
    XS_torch_tst_a = torch.Tensor(testXS[:,:,spatialSplit_allNod.i_tst,:])
    YS_torch_tst_a = torch.Tensor(testYS[:,:,spatialSplit_allNod.i_tst,:])
    print('\ntrain.shape', XS_torch_train.shape, YS_torch_train.shape)
    print('train_model.shape', XS_torch_train_model.shape, YS_torch_train_model.shape)
    print('val_u.shape', XS_torch_val_u.shape, YS_torch_val_u.shape)
    print('val_a.shape', XS_torch_val_a.shape, YS_torch_val_a.shape)
    print('tst_u.shape', XS_torch_tst_u.shape, YS_torch_tst_u.shape)
    print('tst_a.shape', XS_torch_tst_a.shape, YS_torch_tst_a.shape)
    if P.example_verbose:
        print('\nFor veryfication purposes')
        print('Corresponding first sensor in org order in test', XS_torch_tst_u[0,0,1,:])
        print('Corresponding second sensor in org order in train', XS_torch_train[0,0,1,:])
    # torch dataset
    train_data = torch.utils.data.TensorDataset(XS_torch_train, YS_torch_train)
    train_model_data = torch.utils.data.TensorDataset(XS_torch_train_model, YS_torch_train_model)
    val_u_data = torch.utils.data.TensorDataset(XS_torch_val_u, YS_torch_val_u)
    val_a_data = torch.utils.data.TensorDataset(XS_torch_val_a, YS_torch_val_a)
    tst_u_data = torch.utils.data.TensorDataset(XS_torch_tst_u, YS_torch_tst_u)
    tst_a_data = torch.utils.data.TensorDataset(XS_torch_tst_a, YS_torch_tst_a)
    # torch dataloader
    train_iter = torch.utils.data.DataLoader(train_data, P.batch_size, shuffle=False)
    train_model_iter = torch.utils.data.DataLoader(train_model_data, P.batch_size, shuffle=False)
    val_u_iter = torch.utils.data.DataLoader(val_u_data, P.batch_size, shuffle=False)
    val_a_iter = torch.utils.data.DataLoader(val_a_data, P.batch_size, shuffle=False)
    tst_u_iter = torch.utils.data.DataLoader(tst_u_data, P.batch_size, shuffle=False)
    tst_a_iter = torch.utils.data.DataLoader(tst_a_data, P.batch_size, shuffle=False)

    # Load the adjacency matrix
    adj_mx = load_adj(P.adj_path, P.adj_type, P.dataname)
    if P.example_verbose:
        print('\nadjacency matrix after normalization')
        print('Entry (18,1):', adj_mx[0][18][1])
        print('Entry (7,11):', adj_mx[0][7][11])
        print('Entry (19,10):', adj_mx[0][19][10])

    # Split the adjacency matrix by spatial and temporal splits
    adj_train = [torch.tensor(i[spatialSplit_unseen.i_trn,:][:,spatialSplit_unseen.i_trn]).to(device) for i in adj_mx]
    adj_val_u = [torch.tensor(i[spatialSplit_unseen.i_val,:][:,spatialSplit_unseen.i_val]).to(device) for i in adj_mx]
    adj_val_a = [torch.tensor(i[spatialSplit_allNod.i_val,:][:,spatialSplit_allNod.i_val]).to(device) for i in adj_mx]
    adj_tst_u = [torch.tensor(i[spatialSplit_unseen.i_tst,:][:,spatialSplit_unseen.i_tst]).to(device) for i in adj_mx]
    adj_tst_a = [torch.tensor(i[spatialSplit_allNod.i_tst,:][:,spatialSplit_allNod.i_tst]).to(device) for i in adj_mx]
    print('\nadj_train', 'length of', len(adj_train), adj_train[0].shape)
    print('adj_val_u', 'length of', len(adj_val_u), adj_val_u[0].shape)
    print('adj_val_a', 'length of', len(adj_val_a), adj_val_a[0].shape)
    print('adj_tst_u', 'length of', len(adj_tst_u), adj_tst_u[0].shape)
    print('adj_tst_a', 'length of', len(adj_tst_a), adj_tst_a[0].shape)
    
    if P.example_verbose:
        print('\nFor veryfication purposes')
        print('Corresponding Entry (18,1)', adj_train[0][0][1])
        print('Corresponding Entry (7,11)', adj_val_u[0][1][3])
        print('Corresponding Entry (19,10)', adj_train[0][2][4])
    
    '''
    PRETRAIN data loader.
    No need to wrap 12 timestamps into a single instance.
    Therefore, destruct them to one dimension.
    Both of them should have dimension [number_of_sensor, timestamp]
    '''
    pretrn_iter = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(
        XS_torch_train[:,-1,:,0].T), batch_size=1, shuffle=True)
    preval_iter = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(
        torch.tensor(trainXS[:,-1,spatialSplit_allNod.i_val,0]).T.float()),
    batch_size=1, shuffle=False)
    print('\npretrn_iter.dataset.tensors[0].shape', pretrn_iter.dataset.tensors[0].shape)
    print('preval_iter.dataset.tensors[0].shape', preval_iter.dataset.tensors[0].shape)
    if P.example_verbose:
        print('\nFor veryfication purposes')
        a = pretrn_iter.dataset.tensors
        b = preval_iter.dataset.tensors
        print('Corresponding second instance pretrn_iter', a[0][1,:])
        print('Corresponding second instance preval_iter', b[0][1,:])
    return pretrn_iter, preval_iter, spatialSplit_unseen, spatialSplit_allNod, \
        train_iter, train_model_iter, val_u_iter, val_a_iter, tst_u_iter, tst_a_iter, \
        adj_train, adj_val_u, adj_val_a, adj_tst_u, adj_tst_a

def pretrainModel(name, pretrain_iter, preval_iter, adj_train, adj_val, device, spatialSplit_allNod):
    print('pretrainModel Started ...')
    adj_train = [tensor.to(device) for tensor in adj_train]
    adj_val = [tensor.to(device) for tensor in adj_val]
    if P.augmentation == 'sampler':
        is_sampler = True
    else:
        is_sampler = False
    if P.is_cost:
        model = CoSTEncoder(1, 32, P.cost_kernals, P.cost_alpha, P.cl_temperature, P.is_GCN_encoder, is_sampler, len(adj_train)).to(device)
    else:
        model = Contrastive_FeatureExtractor_conv(P.cl_temperature, P.is_GCN_encoder, is_sampler, len(adj_train)).to(device)
    min_val_loss = np.inf
    optimizer = torch.optim.Adam(model.parameters(), lr=P.learn_rate, weight_decay=P.weight_decay)
    s_time = datetime.now()
    for epoch in range(P.pretrain_epoch):
        starttime = datetime.now()
        model.train()
        x = pretrain_iter.dataset.tensors
        optimizer.zero_grad()
        if P.augmentation == 'edge_masking':
            loss = model.contrast(x[0].to(device), x[0].to(device), edge_masking(adj_train, 0.02, device), edge_masking(adj_train, 0.02, device), 0, P.example_verbose)
        elif P.augmentation == 'sampler':
            loss = model.contrast(x[0].to(device), x[0].to(device), adj_train, adj_train, 0, P.example_verbose)
        elif P.augmentation == 'temporal_shifting':
            loss = model.contrast(temporal_shifting(x[0], P.temporal_shifting_r).to(device),temporal_shifting(x[0], P.temporal_shifting_r).to(device), adj_train, adj_train, 0, P.example_verbose)
        loss.backward()
        optimizer.step()
        train_loss = loss / x[0].shape[0]
        if P.is_testunseen:
            sensor_idx_start = len(spatialSplit_allNod.i_trn)
        else:
            sensor_idx_start = 0
        val_loss = pre_evaluateModel(model, preval_iter, adj_val, sensor_idx_start, device)
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            torch.save(model.state_dict(), P.save_path + '/' + name + '.pt')
        endtime = datetime.now()
        epoch_time = (endtime - starttime).seconds
        # save epoch results
        print("epoch", epoch, "time used:", epoch_time," seconds ", "train loss:", train_loss, "validation loss:", val_loss)
        with open(P.save_path + '/' + name + '_log.txt', 'a') as f:
            f.write("%s %d, %s %d %s, %s %.10f, %s %.10f\n" % ("epoch:", epoch, "time used:", epoch_time, "seconds", "train loss:", train_loss, "validation loss:", val_loss))
    e_time = datetime.now()
    print('PRETIME DURATION:', e_time-s_time)
    # Write the final results to the log file
    try:
        df = pd.read_csv('save/results.csv')
    except FileNotFoundError:
        df = pd.DataFrame(columns=['exe_id', 'min_pretrain_val_loss', 'pretrain_time'])
    row_exists = 'exe_id' in df.columns and any(df['exe_id'] == P.exe_id)
    if row_exists:
        df.loc[df['exe_id'] == P.exe_id, ['min_pretrain_val_loss', 'pretrain_time']] = [min_val_loss, e_time-s_time]
    else:
        new_row = pd.DataFrame({'exe_id': [P.exe_id], 'min_pretrain_val_loss': [min_val_loss.cpu().numpy()], 'pretrain_time': [e_time-s_time]})
        df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv('save/results.csv', index=False)
    # End
    
    print('pretrainModel Ended ...\n')

def pre_evaluateModel(model, data_iter, adj, sensor_idx_start, device):
    model.eval()
    with torch.no_grad():
        x = data_iter.dataset.tensors
        if P.augmentation == 'edge_masking':
            l = model.contrast(x[0].to(device), x[0].to(device), edge_masking(adj, 0.02, device), edge_masking(adj, 0.02, device), sensor_idx_start, P.example_verbose)
        elif P.augmentation == 'sampler':
            l = model.contrast(x[0].to(device), x[0].to(device), adj, adj, sensor_idx_start, P.example_verbose)
        elif P.augmentation == 'temporal_shifting':
            l = model.contrast(temporal_shifting(x[0], P.temporal_shifting_r).to(device),temporal_shifting(x[0], P.temporal_shifting_r).to(device), adj, adj, sensor_idx_start, P.example_verbose)
        return l / x[0].shape[0]

def trainModel(name, mode, 
        train_iter, train_model_iter, val_u_iter, val_a_iter,
        adj_train, adj_val_u, adj_val_a,
        spatialSplit_unseen, spatialSplit_allNod, device_cpu, device_gpu):
    print('trainModel Started ...')
    print('TIMESTEP_IN, TIMESTEP_OUT', P.timestep_in, P.timestep_out)
    if P.train_encoder_on == 'cpu':
        device_encoder = device_cpu
    else:
        device_encoder = device_gpu

    if P.augmentation == 'sampler':
        is_sampler = True
    else:
        is_sampler = False

    model = getModel(name, device_gpu, len(adj_train))
    min_val_u_loss = np.inf
    min_val_a_loss = np.inf
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=P.learn_rate, weight_decay=P.weight_decay)
    s_time = datetime.now()
    adj_train = [tensor.to(device_encoder) for tensor in adj_train]
    adj_val_u = [tensor.to(device_encoder) for tensor in adj_val_u]
    adj_val_a = [tensor.to(device_encoder) for tensor in adj_val_a]
    if P.is_pretrain:
        if P.is_cost:
            encoder = CoSTEncoder(1, 32, P.cost_kernals, P.cost_alpha, P.cl_temperature, P.is_GCN_encoder, is_sampler, len(adj_train)).to(device_encoder)
        else:
            encoder = Contrastive_FeatureExtractor_conv(P.cl_temperature, P.is_GCN_encoder, is_sampler, len(adj_train)).to(device_encoder)
        encoder.eval()
        with torch.no_grad():
            encoder.load_state_dict(torch.load(P.save_path+ '/' + 'encoder' + '.pt'))
            train_embed = encoder(train_iter.dataset.tensors[0][:,-1,:,0].T.to(device_encoder), adj_train, P.example_verbose).T.detach().to(device_gpu)
            val_u_embed = encoder(torch.Tensor(data[:P.train_size,spatialSplit_unseen.i_val]).to(device_encoder).float().T, adj_val_u, P.example_verbose).T.detach().to(device_gpu)
            val_a_embed = encoder(torch.Tensor(data[:P.train_size,spatialSplit_allNod.i_val]).to(device_encoder).float().T, adj_val_a, P.example_verbose).T.detach().to(device_gpu)
    else:
        train_embed = torch.zeros(32, train_iter.dataset.tensors[0].shape[2]).to(device_gpu).detach()
        val_u_embed = torch.zeros(32, val_u_iter.dataset.tensors[0].shape[2]).to(device_gpu).detach()
        val_a_embed = torch.zeros(32, val_a_iter.dataset.tensors[0].shape[2]).to(device_gpu).detach()
    adj_train = [tensor.to(device_gpu) for tensor in adj_train]
    adj_val_u = [tensor.to(device_gpu) for tensor in adj_val_u]
    adj_val_a = [tensor.to(device_gpu) for tensor in adj_val_a]
    m_time = datetime.now()
    print('ENCODER INFER DURATION IN MODEL TRAINING:', m_time-s_time)
    print('train_embed', train_embed.shape, train_embed.mean(), train_embed.std())
    print('val_u_embed', val_u_embed.shape, val_u_embed.mean(), val_u_embed.std())
    print('val_a_embed', val_a_embed.shape, val_a_embed.mean(), val_a_embed.std())
    if P.train_model_datasplit == 'A':
        model_input = train_iter
    elif P.train_model_datasplit == 'B':
        model_input = train_model_iter
    for epoch in range(P.train_epoch):
        starttime = datetime.now()     
        loss_sum, n = 0.0, 0
        model.train()
        for x, y in model_input:
            optimizer.zero_grad()
            if P.model == 'gwnet':
                y_pred = model(x.to(device_gpu), adj_train, train_embed)
            elif P.model == 'LSTM':
                y_pred = model(x.to(device_gpu), train_embed, P.encoder_to_model_ratio, P.is_concat_encoder_model, support = adj_train)
            loss = criterion(y_pred, y.to(device_gpu))
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * y.shape[0]
            n += y.shape[0]
        train_loss = loss_sum / n
        if P.is_testunseen:
            sensor_idx_start = len(spatialSplit_allNod.i_trn)
        else:
            sensor_idx_start = 0
        val_u_loss = evaluateModel(model, criterion, val_u_iter, adj_val_u, val_u_embed, device_gpu, 0)
        val_a_loss = evaluateModel(model, criterion, val_a_iter, adj_val_a, val_a_embed, device_gpu, sensor_idx_start)
        if val_u_loss < min_val_u_loss:
            min_val_u_loss = val_u_loss
            torch.save(model.state_dict(), P.save_path + '/' + name + '_u.pt')
        if val_a_loss < min_val_a_loss:
            min_val_a_loss = val_a_loss
            torch.save(model.state_dict(), P.save_path + '/' + name + '_a.pt')
        endtime = datetime.now()
        epoch_time = (endtime - starttime).seconds
        print("epoch", epoch,
            "time used:",epoch_time," seconds ",
            "train loss:", train_loss,
            "validation unseen nodes loss:", val_u_loss,
            "validation all nodes loss:", val_a_loss)
        with open(P.save_path + '/' + name + '_log.txt', 'a') as f:
            f.write("%s %d, %s %d %s, %s %.10f, %s %.10f, %s %.10f\n" % \
                ("epoch", epoch,
                 "time used:",epoch_time," seconds ",
                 "train loss:", train_loss,
                 "validation unseen nodes loss:", val_u_loss,
                 "validation all nodes loss:", val_a_loss))
    e_time = datetime.now()
    print('MODEL TRAINING DURATION:', e_time-m_time)
    torch_score = evaluateModel(model, criterion, model_input, adj_train, train_embed, device_gpu, 0)
    # Write the final results to the log file
    df = pd.read_csv('save/results.csv')
    row_exists = 'exe_id' in df.columns and any(df['exe_id'] == P.exe_id)
    if row_exists:
        df.loc[df['exe_id'] == P.exe_id, ['train_loss', 'min_val_a_loss', 'min_val_u_loss', 'train_time']] = [torch_score, min_val_a_loss, min_val_u_loss, e_time-m_time]
    else:
        new_row = pd.DataFrame({'exe_id': [P.exe_id], 'train_loss': [torch_score], 'min_val_a_loss': [min_val_a_loss], 'min_val_u_loss': [min_val_u_loss], 'train_time': [e_time-m_time]})
        df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv('save/results.csv', index=False)
    with open(P.save_path + '/' + name + '_prediction_scores.txt', 'a') as f:
        f.write("%s, %s, %s, %.10e, %.10f\n" % (name, mode, 'MAE on train', torch_score, torch_score))
    print('trainModel Ended ...\n')

def evaluateModel(model, criterion, data_iter, adj, embed, device, sensor_idx_start):
    model.eval()
    torch.cuda.empty_cache()
    l_sum, n = 0.0, 0
    with torch.no_grad():
        for x, y in data_iter:
            if P.model == 'gwnet':
                y_pred = model(x.to(device), adj, embed)
            elif P.model == 'LSTM':
                y_pred = model(x.to(device), embed, P.encoder_to_model_ratio, P.is_concat_encoder_model, support = adj)
            y_pred = y_pred[:,:,sensor_idx_start:,]
            y = y[:,:,sensor_idx_start:,]
            l = criterion(y_pred, y.to(device))
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
    return l_sum / n

def testModel(name, mode, test_iter, adj_tst, spatialsplit, device_cpu, device_gpu):
    criterion = nn.L1Loss()
    print('Model Testing', mode, 'Started ...')
    if P.train_encoder_on == 'cpu':
        device_encoder = device_cpu
    else:
        device_encoder = device_gpu

    if P.augmentation == 'sampler':
        is_sampler = True
    else:
        is_sampler = False
    
    if P.is_pretrain:
        if P.is_cost:
            encoder = CoSTEncoder(1, 32, P.cost_kernals, P.cost_alpha, P.cl_temperature, P.is_GCN_encoder, is_sampler, len(adj_tst)).to(device_encoder)
        else:
            encoder = Contrastive_FeatureExtractor_conv(P.cl_temperature, P.is_GCN_encoder, is_sampler, len(adj_tst)).to(device_encoder)
        encoder.load_state_dict(torch.load(P.save_path+ '/' + 'encoder' + '.pt'))
        encoder.eval()
    model = getModel(name, device_gpu, len(adj_tst))
    model.load_state_dict(torch.load(P.save_path+ '/' + name +mode[-2:]+ '.pt'))
    s_time = datetime.now()
    
    print('Model Infer Start ...')
    tst_embed = torch.zeros(32, test_iter.dataset.tensors[0].shape[2]).to(device_gpu).detach()
    adj_tst = [tensor.to(device_encoder) for tensor in adj_tst]
    if P.is_pretrain:
        with torch.no_grad():
            tst_embed = encoder(torch.Tensor(data[:P.trainval_size,spatialsplit.i_tst]).to(device_encoder).float().T, adj_tst, P.example_verbose).T.detach().to(device_gpu)
    adj_tst = [tensor.to(device_gpu) for tensor in adj_tst]
    m_time = datetime.now()
    print('ENCODER INFER DURATION:', m_time-s_time)
    if P.is_testunseen:
        sensor_idx_start = len(spatialsplit.i_val)
    else:
        sensor_idx_start = 0
    torch_score = evaluateModel(model, criterion, test_iter, adj_tst, tst_embed, device_gpu, sensor_idx_start)
    e_time = datetime.now()
    print('Model Infer End ...', e_time)
    
    print('MODEL INFER DURATION:', e_time-m_time)
    YS_pred = predictModel(model, test_iter, adj_tst, tst_embed, device_gpu)
    YS = test_iter.dataset.tensors[1].cpu().numpy()
    if P.is_testunseen:
        YS_pred = YS_pred[:,:,len(spatialsplit.i_val):,:]
        YS = YS[:,:,len(spatialsplit.i_val):,:]
    print('YS.shape, YS_pred.shape,', YS.shape, YS_pred.shape)
    original_shape = np.squeeze(YS).shape
    YS = scaler.inverse_transform(np.squeeze(YS).reshape(-1, YS.shape[2])).reshape(original_shape)
    YS_pred  = scaler.inverse_transform(np.squeeze(YS_pred).reshape(-1, YS_pred.shape[2])).reshape(original_shape)
    # print('YS.shape, YS_pred.shape,', YS.shape, YS_pred.shape)
    # np.save(P.save_path + '/' + P.MODELNAME + '_' + mode + '_' + name +'_prediction.npy', YS_pred)
    # np.save(P.save_path + '/' + P.MODELNAME + '_' + mode + '_' + name +'_groundtruth.npy', YS)
    MSE, RMSE, MAE, MAPE = Metrics.evaluate(YS, YS_pred)
    # Write the final results to the log file
    df = pd.read_csv('save/results.csv')
    columns_to_update = [mode+'_loss', 'encoder_infer_time', 'model_infer_time']
    values_to_assign = [torch_score, m_time-s_time, e_time-m_time]
    df.loc[df['exe_id'] == P.exe_id, columns_to_update] = values_to_assign
    df.to_csv('save/results.csv', index=False) 

    print('*' * 40)
    print("%s, %s, Torch MSE, %.10e, %.10f" % (name, mode, torch_score, torch_score))
    f = open(P.save_path + '/' + name + '_prediction_scores.txt', 'a')
    f.write("%s, %s, Torch MSE, %.10e, %.10f\n" % (name, mode, torch_score, torch_score))
    print("all pred steps, %s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f" % (name, mode, MSE, RMSE, MAE, MAPE))
    f.write("all pred steps, %s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f\n" % (name, mode, MSE, RMSE, MAE, MAPE))
    for i in range(P.timestep_out):
        MSE, RMSE, MAE, MAPE = Metrics.evaluate(YS[:, i, :], YS_pred[:, i, :])
        print("%d step, %s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f" % (i+1, name, mode, MSE, RMSE, MAE, MAPE))
        f.write("%d step, %s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f\n" % (i+1, name, mode, MSE, RMSE, MAE, MAPE))
    f.close()
    print('Model Testing Ended ...', time.ctime())

def predictModel(model, data_iter, adj, embed, device):
    YS_pred = []
    model.eval()
    with torch.no_grad():
        for x, y in data_iter:
            if P.model == 'gwnet':
                YS_pred_batch = model(x.to(device), adj, embed)
            elif P.model == 'LSTM':
                YS_pred_batch = model(x.to(device), embed, P.encoder_to_model_ratio, P.is_concat_encoder_model, support = adj)
            YS_pred_batch = YS_pred_batch.cpu().numpy()
            YS_pred.append(YS_pred_batch)
        YS_pred = np.vstack(YS_pred)
    return YS_pred

'''
System parameters
P = type('Parameters', (object,), {})()
P.dataname = 'METRLA'
P.model = 'LSTM'
P.seed = 0

P.t_train = 0.7
P.t_val = 0.1
P.s_train = 0.7
P.s_val = 0.1

P.timestep_in = 12
P.timestep_out = 12
P.n_channel = 1
P.batch_size = 64

P.lstm_hidden_dim = 128
P.gwnet_is_adp_adj = True
P.gwnet_is_SGA = False

P.adj_type = 'doubletransition'
P.is_cost = True
P.cost_kernals = [1, 2, 4, 8, 16, 32, 64, 128]
P.cost_alpha = 0.5
P.cl_temperature = 1
P.is_pretrain = True
P.is_GCN_encoder = True
P.is_GCN_after_CL = True
P.augmentation = 'sampler'
P.temporal_shifting_r = 0.8
P.encoder_to_model_ratio = 1
P.is_concat_encoder_model = True

P.learn_rate = 0.001
P.pretrain_epoch = 2
P.train_epoch = 1
P.weight_decay = 0
P.is_testunseen = True
P.train_model_datasplit = 'B'
P.train_encoder_on = 'cpu'

P.example_verbose = True

Not possible: gcn false and sampler false
'''

def main():
    if P.is_pretrain == False:
        P.is_concat_encoder_model = False
    global data
    global scaler

    '''
    Set backend devices. 
    Check the type of GPU, either mps or cuda, is available.
    Also, another options is to use the CPU.
    '''
    if torch.backends.mps.is_available():
        device_gpu = torch.device('mps') 
    if torch.cuda.is_available():
        device_gpu = torch.device('cuda')
    device_cpu = torch.device('cpu')

    # load data from file
    P.exe_id = P.dataname + '_' + datetime.now().strftime("%y%m%d-%H%M")
    P.save_path = 'save/' + P.exe_id
    if P.dataname == 'METRLA':
        print('P.dataname == METRLA')
        P.data_path = './data/METRLA/metr-la.h5'
        P.adj_path = './data/METRLA/adj_mx_new.pkl'
        P.n_sensor = 207
        data = pd.read_hdf(P.data_path).values
    elif P.dataname == 'PEMSBAY':
        print('P.dataname == PEMSBAY')
        P.data_path = './data/PEMSBAY/pems-bay.h5'
        P.adj_path = './data/PEMSBAY/adj_mx_new.pkl'
        P.n_sensor = 325
        data = pd.read_hdf(P.data_path).values
    elif P.dataname == 'HAGUE':
        print('P.dataname == HAGUE')
        P.data_path = './data/Hauge/hague_comp_filled.h5'
        P.adj_path = './data/Hauge/adj_mx_comp.pkl'
        P.n_sensor = 144
        data = pd.read_hdf(P.data_path).values
    elif P.dataname == 'HAGUE_75':
        print('P.dataname == HAGUE_75')
        P.data_path = './data/Hauge/hague_filled_75.h5'
        P.adj_path = './data/Hauge/adj_mx.pkl'
        P.n_sensor = 144
        data = pd.read_hdf(P.data_path).values
    elif P.dataname == 'EXAMPLE':
        print('P.dataname == EXAMPLE')
        P.data_path = './data/example.h5'
        P.adj_path = './data/adj_mx_ex.pkl'
        P.n_sensor = 20
        data = pd.read_hdf(P.data_path).values
        if P.example_verbose:
            print('\nFirst row at 10 am:', data[0])
            print('\nSecond row at 10:05:', data[1])
    
    '''
    Apply the scaler to the sensor readings. 
    The data is scaled based on the mean and standard deviation of the data.
    '''
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    if P.example_verbose:
        print('data.shape:', data.shape)
        if P.dataname == 'EXAMPLE':
            print('\nFirst row at 10 am after scaling:', data[0])
            print('\nSecond row at 10:05 after scaling:', data[1])
            print('\nFirst sensor data:', data[:,0])
            print('\nSecond sensor data:', data[:,1])

    # setup
    pretrn_iter, preval_iter, spatialSplit_unseen, spatialSplit_allNod, \
        train_iter, train_model_iter, val_u_iter, val_a_iter, tst_u_iter, tst_a_iter, \
        adj_train, adj_val_u, adj_val_a, adj_tst_u, adj_tst_a = setups(device_gpu)
    
    # save parameters
    save_parameters(P, 'save/parameters.csv')
    
    if P.is_pretrain:
        if P.train_encoder_on == 'cpu':
            pretrainModel('encoder', pretrn_iter, preval_iter, adj_train, adj_val_a, device_cpu, spatialSplit_allNod)
        else:
            pretrainModel('encoder', pretrn_iter, preval_iter, adj_train, adj_val_a, device_gpu, spatialSplit_allNod)

    trainModel(P.model, 'train',
        train_iter, train_model_iter, val_u_iter, val_a_iter,
        adj_train, adj_val_u, adj_val_a,
        spatialSplit_unseen, spatialSplit_allNod, device_cpu, device_gpu)
    
    testModel(P.model, 'test_a', tst_a_iter, adj_tst_a, spatialSplit_allNod, device_cpu, device_gpu)


if __name__ == '__main__':
    main()

    