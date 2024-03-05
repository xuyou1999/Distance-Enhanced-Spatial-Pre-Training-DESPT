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

def save_parameters(param_obj, filename):
    with open(filename, "w") as file:
        for attr in dir(param_obj):
            # Filter out built-in attributes/methods
            if not attr.startswith("__"):
                value = getattr(param_obj, attr)
                file.write(f"{attr} = {value}\n")

def getModel(name, device):
    if name == 'gwnet':
        model = gwnet(device, num_nodes=P.N_NODE, in_dim=P.CHANNEL, adp_adj=P.is_adp_adj, sga=P.is_SGA).to(device)
    elif name == 'LSTM':
        model = LSTM_uni(input_dim=P.CHANNEL, hidden_dim=P.hidden_dim, device=device).to(device)
    return model

def getXSYS(data, mode):
    TRAIN_NUM = int(data.shape[0] * (P.T_TRN + P.T_VAL))
    XS, YS = [], []
    if mode == 'TRAIN':    
        for i in range(TRAIN_NUM - P.TIMESTEP_OUT - P.TIMESTEP_IN + 1):
            x = data[i:i+P.TIMESTEP_IN, :]
            y = data[i+P.TIMESTEP_IN:i+P.TIMESTEP_IN+P.TIMESTEP_OUT, :]
            XS.append(x), YS.append(y)
    elif mode == 'TEST':
        for i in range(TRAIN_NUM - P.TIMESTEP_IN,  data.shape[0] - P.TIMESTEP_OUT - P.TIMESTEP_IN + 1):
            x = data[i:i+P.TIMESTEP_IN, :]
            y = data[i+P.TIMESTEP_IN:i+P.TIMESTEP_IN+P.TIMESTEP_OUT, :]
            XS.append(x), YS.append(y)
    XS, YS = np.array(XS), np.array(YS)
    XS, YS = XS[:, :, :, np.newaxis], YS[:, :, :, np.newaxis]
    XS = XS.transpose(0, 3, 2, 1)
    return XS, YS

def pre_evaluateModel(model, data_iter, adj, sensor_idx_start):
    model.eval()
    with torch.no_grad():
        x = data_iter.dataset.tensors
        if P.is_GCN == True and P.is_sampler == False:
            l = model.contrast(x[0].to(device), edge_masking(adj, 0.02, device), edge_masking(adj, 0.02, device), sensor_idx_start)
        else:
            l = model.contrast(x[0].to(device), adj, adj, sensor_idx_start)
        return l / x[0].shape[0]

def evaluateModel(model, criterion, data_iter, adj, embed, sensor_idx_start=0):
    model.eval()
    torch.cuda.empty_cache()
    l_sum, n = 0.0, 0
    with torch.no_grad():
        for x, y in data_iter:
            if P.MODEL == 'gwnet':
                y_pred = model(x.to(device), adj, embed)
            elif P.MODEL == 'LSTM':
                y_pred = model(x.to(device), embed)
            y_pred = y_pred[:,:,sensor_idx_start:,]
            y = y[:,:,sensor_idx_start:,]
            # print('y_pred.shape', y_pred.shape)
            # print('y.shape', y.shape)
            l = criterion(y_pred, y.to(device))
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
    return l_sum / n

def predictModel(model, data_iter, adj, embed):
    YS_pred = []
    model.eval()
    with torch.no_grad():
        for x, y in data_iter:
            if P.MODEL == 'gwnet':
                YS_pred_batch = model(x.to(device), adj, embed)
            elif P.MODEL == 'LSTM':
                YS_pred_batch = model(x.to(device), embed)
            YS_pred_batch = YS_pred_batch.cpu().numpy()
            YS_pred.append(YS_pred_batch)
        YS_pred = np.vstack(YS_pred)
    return YS_pred

def setups():
    # make save folder
    if not os.path.exists(P.PATH):
        os.makedirs(P.PATH)
    # seed
    torch.manual_seed(P.seed)
    torch.cuda.manual_seed(P.seed)
    np.random.seed(P.seed)
    print(P.KEYWORD, 'data splits', time.ctime())
    # test split temporal
    trainXS, trainYS = getXSYS(data, 'TRAIN')
    testXS, testYS = getXSYS(data, 'TEST')

    print('trainXS.shape', trainXS.shape)
    print('trainYS.shape', trainYS.shape)
    print('testXS.shape', testXS.shape)
    print('testYS.shape', testYS.shape)
    # trn val split
    P.trainval_size = len(trainXS)
    P.train_size = int(P.trainval_size * (P.T_TRN / (P.T_TRN + P.T_VAL)))
    XS_torch_trn = trainXS[:P.train_size,:,:,:]
    YS_torch_trn = trainYS[:P.train_size,:,:,:]
    XS_torch_val = trainXS[P.train_size:P.trainval_size,:,:,:]
    YS_torch_val = trainYS[P.train_size:P.trainval_size,:,:,:]

    # # spatial split
    spatialSplit_unseen = unseen_nodes.SpatialSplit(data.shape[1], r_trn=P.S_TRN, r_val=.1, r_tst=.2, seed=P.seed)
    spatialSplit_allNod = unseen_nodes.SpatialSplit(data.shape[1], r_trn=P.S_TRN, r_val=min(1.0,P.S_VAL+P.S_TRN), r_tst=1.0, seed=P.seed)
    print('spatialSplit_unseen', spatialSplit_unseen)
    print('spatialSplit_allNod', spatialSplit_allNod)
    # print(spatialSplit_allNod.i_trn)
    # print(spatialSplit_allNod.i_val)
    # print(spatialSplit_unseen.i_tst)
    XS_torch_train = torch.Tensor(XS_torch_trn[:,:,spatialSplit_unseen.i_trn,:])
    YS_torch_train = torch.Tensor(YS_torch_trn[:,:,spatialSplit_unseen.i_trn,:])
    XS_torch_val_u = torch.Tensor(XS_torch_val[:,:,spatialSplit_unseen.i_val,:])
    YS_torch_val_u = torch.Tensor(YS_torch_val[:,:,spatialSplit_unseen.i_val,:])
    XS_torch_val_a = torch.Tensor(XS_torch_val[:,:,spatialSplit_allNod.i_val,:])
    YS_torch_val_a = torch.Tensor(YS_torch_val[:,:,spatialSplit_allNod.i_val,:])
    XS_torch_tst_u = torch.Tensor(testXS[:,:,spatialSplit_unseen.i_tst,:])
    YS_torch_tst_u = torch.Tensor(testYS[:,:,spatialSplit_unseen.i_tst,:])
    XS_torch_tst_a = torch.Tensor(testXS[:,:,spatialSplit_allNod.i_tst,:])
    YS_torch_tst_a = torch.Tensor(testYS[:,:,spatialSplit_allNod.i_tst,:])
    print('train.shape', XS_torch_train.shape, YS_torch_train.shape)
    print('val_u.shape', XS_torch_val_u.shape, YS_torch_val_u.shape)
    print('val_a.shape', XS_torch_val_a.shape, YS_torch_val_a.shape)
    print('tst_u.shape', XS_torch_tst_u.shape, YS_torch_tst_u.shape)
    print('tst_a.shape', XS_torch_tst_a.shape, YS_torch_tst_a.shape)
    # torch dataset
    train_data = torch.utils.data.TensorDataset(XS_torch_train, YS_torch_train)
    val_u_data = torch.utils.data.TensorDataset(XS_torch_val_u, YS_torch_val_u)
    val_a_data = torch.utils.data.TensorDataset(XS_torch_val_a, YS_torch_val_a)
    tst_u_data = torch.utils.data.TensorDataset(XS_torch_tst_u, YS_torch_tst_u)
    tst_a_data = torch.utils.data.TensorDataset(XS_torch_tst_a, YS_torch_tst_a)
    # torch dataloader
    train_iter = torch.utils.data.DataLoader(train_data, P.BATCHSIZE, shuffle=True)
    val_u_iter = torch.utils.data.DataLoader(val_u_data, P.BATCHSIZE, shuffle=False)
    val_a_iter = torch.utils.data.DataLoader(val_a_data, P.BATCHSIZE, shuffle=False)
    tst_u_iter = torch.utils.data.DataLoader(tst_u_data, P.BATCHSIZE, shuffle=False)
    tst_a_iter = torch.utils.data.DataLoader(tst_a_data, P.BATCHSIZE, shuffle=False)
    # adj matrix spatial split
    adj_mx = load_adj(P.ADJPATH, None, P.DATANAME)
    adj_train = [torch.tensor(i[spatialSplit_unseen.i_trn,:][:,spatialSplit_unseen.i_trn]).to(device) for i in adj_mx]
    adj_val_u = [torch.tensor(i[spatialSplit_unseen.i_val,:][:,spatialSplit_unseen.i_val]).to(device) for i in adj_mx]
    adj_val_a = [torch.tensor(i[spatialSplit_allNod.i_val,:][:,spatialSplit_allNod.i_val]).to(device) for i in adj_mx]
    adj_tst_u = [torch.tensor(i[spatialSplit_unseen.i_tst,:][:,spatialSplit_unseen.i_tst]).to(device) for i in adj_mx]
    adj_tst_a = [torch.tensor(i[spatialSplit_allNod.i_tst,:][:,spatialSplit_allNod.i_tst]).to(device) for i in adj_mx]
    print('adj_train', len(adj_train), adj_train[0].shape)
    print('adj_val_u', len(adj_val_u), adj_val_u[0].shape)
    print('adj_val_a', len(adj_val_a), adj_val_a[0].shape)
    print('adj_tst_u', len(adj_tst_u), adj_tst_u[0].shape)
    print('adj_tst_a', len(adj_tst_a), adj_tst_a[0].shape)
    # PRETRAIN data loader
    pretrn_iter = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(
        XS_torch_train[:,-1,:,0].T), batch_size=1, shuffle=True)
    preval_iter = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(
        torch.tensor(trainYS[:,-1,spatialSplit_allNod.i_val,0]).T.float()),
    batch_size=1, shuffle=False)
    print('pretrn_iter.dataset.tensors[0].shape', pretrn_iter.dataset.tensors[0].shape)
    print('preval_iter.dataset.tensors[0].shape', preval_iter.dataset.tensors[0].shape)
    return pretrn_iter, preval_iter, spatialSplit_unseen, spatialSplit_allNod, \
        train_iter, val_u_iter, val_a_iter, tst_u_iter, tst_a_iter, \
        adj_train, adj_val_u, adj_val_a, adj_tst_u, adj_tst_a

def pretrainModel(name, pretrain_iter, preval_iter, adj_train, adj_val_u, device, spatialSplit_allNod):
    print('pretrainModel Started ...', time.ctime())
    model = Contrastive_FeatureExtractor_conv(P.TEMPERATURE, P.is_GCN, P.is_sampler).to(device)
    min_val_loss = np.inf
    optimizer = torch.optim.Adam(model.parameters(), lr=P.LEARN, weight_decay=P.weight_decay)
    s_time = datetime.now()
    for epoch in range(P.PRETRN_EPOCH):
        starttime = datetime.now()
        model.train()
        x = pretrain_iter.dataset.tensors
        optimizer.zero_grad()
        if P.is_GCN == True and P.is_sampler == False:
            loss = model.contrast(x[0].to(device), edge_masking(adj_train, 0.02, device), edge_masking(adj_train, 0.02, device), 0, len(x[0]))
        else:
            loss = model.contrast(x[0].to(device), adj_train, adj_train, 0, len(x[0]))
        loss.backward()
        optimizer.step()
        train_loss = loss / x[0].shape[0]
        if P.is_testunseen:
            sensor_idx_start = len(spatialSplit_allNod.i_trn)
        else:
            sensor_idx_start = 0
        val_loss = pre_evaluateModel(model, preval_iter, adj_val_u, sensor_idx_start, 'last')
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            torch.save(model.state_dict(), P.PATH + '/' + name + '.pt')
        endtime = datetime.now()
        epoch_time = (endtime - starttime).seconds
        print("epoch", epoch, "time used:", epoch_time," seconds ", "train loss:", train_loss, "validation loss:", val_loss)
        with open(P.PATH + '/' + name + '_log.txt', 'a') as f:
            f.write("%s, %d, %s, %d, %s, %s, %.10f, %s, %.10f\n" % ("epoch", epoch, "time used", epoch_time, "seconds", "train loss", train_loss, "validation loss:", val_loss))
    e_time = datetime.now()
    print('PRETIME DURATION:', e_time, '-', s_time, '=', e_time-s_time)
    print('pretrainModel Ended ...', time.ctime())

def trainModel(name, mode, 
        train_iter, val_u_iter, val_a_iter,
        adj_train, adj_val_u, adj_val_a,
        spatialSplit_unseen, spatialSplit_allNod):
    print('trainModel Started ...', time.ctime())
    print('TIMESTEP_IN, TIMESTEP_OUT', P.TIMESTEP_IN, P.TIMESTEP_OUT)
    model = getModel(name, device)
    min_val_u_loss = np.inf
    min_val_a_loss = np.inf
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=P.LEARN, weight_decay=P.weight_decay)
    s_time = datetime.now()
    print('Model Training Started ...', s_time)
    if P.IS_PRETRN:
        encoder = Contrastive_FeatureExtractor_conv(P.TEMPERATURE, P.is_GCN, P.is_sampler).to(device)
        encoder.eval()
        with torch.no_grad():
            encoder.load_state_dict(torch.load(P.PATH+ '/' + 'encoder' + '.pt'))
            train_embed = encoder(train_iter.dataset.tensors[0][:,-1,:,0].T.to(device), adj_train).T.detach()
            val_u_embed = encoder(torch.Tensor(data[:P.train_size,spatialSplit_unseen.i_val]).to(device).float().T, adj_val_u).T.detach()
            val_a_embed = encoder(torch.Tensor(data[:P.train_size,spatialSplit_allNod.i_val]).to(device).float().T, adj_val_a).T.detach()
    else:
        train_embed = torch.zeros(32, train_iter.dataset.tensors[0].shape[2]).to(device).detach()
        val_u_embed = torch.zeros(32, val_u_iter.dataset.tensors[0].shape[2]).to(device).detach()
        val_a_embed = torch.zeros(32, val_a_iter.dataset.tensors[0].shape[2]).to(device).detach()
    m_time = datetime.now()
    print('ENCODER INFER DURATION IN MODEL TRAINING:', m_time, '-', s_time, '=', m_time-s_time)
    print('train_embed', train_embed.shape, train_embed.mean(), train_embed.std())
    print('val_u_embed', val_u_embed.shape, val_u_embed.mean(), val_u_embed.std())
    print('val_a_embed', val_a_embed.shape, val_a_embed.mean(), val_a_embed.std())
    for epoch in range(P.EPOCH):
        starttime = datetime.now()     
        loss_sum, n = 0.0, 0
        model.train()
        for x, y in train_iter:
            optimizer.zero_grad()
            # print('x.shape, emb', x.shape, train_embed.shape)
            if P.MODEL == 'gwnet':
                y_pred = model(x.to(device), adj_train, train_embed)
            elif P.MODEL == 'LSTM':
                y_pred = model(x.to(device), train_embed)
            loss = criterion(y_pred, y.to(device))
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * y.shape[0]
            n += y.shape[0]
        train_loss = loss_sum / n
        if P.is_testunseen:
            sensor_idx_start = len(spatialSplit_allNod.i_trn)
        else:
            sensor_idx_start = 0
        val_u_loss = evaluateModel(model, criterion, val_u_iter, adj_val_u, val_u_embed, 0)
        val_a_loss = evaluateModel(model, criterion, val_a_iter, adj_val_a, val_a_embed, sensor_idx_start)
        if val_u_loss < min_val_u_loss:
            min_val_u_loss = val_u_loss
            torch.save(model.state_dict(), P.PATH + '/' + name + '_u.pt')
        if val_a_loss < min_val_a_loss:
            min_val_a_loss = val_a_loss
            torch.save(model.state_dict(), P.PATH + '/' + name + '_a.pt')
        endtime = datetime.now()
        epoch_time = (endtime - starttime).seconds
        print("epoch", epoch,
            "time used:",epoch_time," seconds ",
            "train loss:", train_loss,
            "validation unseen nodes loss:", val_u_loss,
            "validation all nodes loss:", val_a_loss)
        with open(P.PATH + '/' + name + '_log.txt', 'a') as f:
            f.write("%s, %d, %s, %d, %s, %s, %.10f, %s, %.10f, %s, %.10f\n" % \
                ("epoch", epoch,
                 "time used:",epoch_time," seconds ",
                 "train loss:", train_loss,
                 "validation unseen nodes loss:", val_u_loss,
                 "validation all nodes loss:", val_a_loss))
    e_time = datetime.now()
    print('MODEL TRAINING DURATION:', e_time, '-', s_time, '=', e_time-s_time)
    torch_score = evaluateModel(model, criterion, train_iter, adj_train, train_embed, 0)
    with open(P.PATH + '/' + name + '_prediction_scores.txt', 'a') as f:
        f.write("%s, %s, %s, %.10e, %.10f\n" % (name, mode, 'MAE on train', torch_score, torch_score))
    # print('*' * 40)
    # print("%s, %s, %s, %.10e, %.10f" % (name, mode, 'MAE on train', torch_score, torch_score))
    # print('min_val_u_loss', min_val_u_loss)
    # print('min_val_a_loss', min_val_a_loss)
    print('trainModel Ended ...', time.ctime())

def testModel(name, mode, test_iter, adj_tst, spatialsplit):
    criterion = nn.L1Loss()
    print('Model Testing', mode, 'Started ...', time.ctime())
    print('TIMESTEP_IN, TIMESTEP_OUT', P.TIMESTEP_IN, P.TIMESTEP_OUT)
    encoder = Contrastive_FeatureExtractor_conv(P.TEMPERATURE, P.is_GCN, P.is_sampler).to(device)
    encoder.load_state_dict(torch.load(P.PATH+ '/' + 'encoder' + '.pt'))
    encoder.eval()
    model = getModel(name, device)
    model.load_state_dict(torch.load(P.PATH+ '/' + name +mode[-2:]+ '.pt'))
    s_time = datetime.now()
    
    print('Model Infer Start ...', s_time)
    tst_embed = torch.zeros(32, test_iter.dataset.tensors[0].shape[2]).to(device).detach()
    if P.IS_PRETRN:
        with torch.no_grad():
            tst_embed = encoder(torch.Tensor(data[:P.trainval_size,spatialsplit.i_tst]).to(device).float().T, adj_tst).T.detach()

    m_time = datetime.now()
    print('ENCODER INFER DURATION:', m_time, '-', s_time, '=', m_time-s_time)
    if P.is_testunseen:
        sensor_idx_start = len(spatialsplit.i_val)
    else:
        sensor_idx_start = 0
    torch_score = evaluateModel(model, criterion, test_iter, adj_tst, tst_embed, sensor_idx_start)
    e_time = datetime.now()
    print('Model Infer End ...', e_time)
    
    print('MODEL INFER DURATION:', e_time, '-', s_time, '=', e_time-s_time)
    YS_pred = predictModel(model, test_iter, adj_tst, tst_embed)
    YS = test_iter.dataset.tensors[1].cpu().numpy()
    if P.is_testunseen:
        YS_pred = YS_pred[:,:,len(spatialsplit.i_val):,:]
        YS = YS[:,:,len(spatialsplit.i_val):,:]
    print('YS.shape, YS_pred.shape,', YS.shape, YS_pred.shape)
    # original_shape = np.squeeze(YS).shape
    # YS = scaler.inverse_transform(np.squeeze(YS).reshape(-1, YS.shape[2])).reshape(original_shape)
    # YS_pred  = scaler.inverse_transform(np.squeeze(YS_pred).reshape(-1, YS_pred.shape[2])).reshape(original_shape)
    # print('YS.shape, YS_pred.shape,', YS.shape, YS_pred.shape)
    # np.save(P.PATH + '/' + P.MODELNAME + '_' + mode + '_' + name +'_prediction.npy', YS_pred)
    # np.save(P.PATH + '/' + P.MODELNAME + '_' + mode + '_' + name +'_groundtruth.npy', YS)
    MSE, RMSE, MAE, MAPE = Metrics.evaluate(YS, YS_pred)
    print('*' * 40)
    print("%s, %s, Torch MSE, %.10e, %.10f" % (name, mode, torch_score, torch_score))
    f = open(P.PATH + '/' + name + '_prediction_scores.txt', 'a')
    f.write("%s, %s, Torch MSE, %.10e, %.10f\n" % (name, mode, torch_score, torch_score))
    print("all pred steps, %s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f" % (name, mode, MSE, RMSE, MAE, MAPE))
    f.write("all pred steps, %s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f\n" % (name, mode, MSE, RMSE, MAE, MAPE))
    for i in range(P.TIMESTEP_OUT):
        MSE, RMSE, MAE, MAPE = Metrics.evaluate(YS[:, i, :], YS_pred[:, i, :])
        print("%d step, %s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f" % (i+1, name, mode, MSE, RMSE, MAE, MAPE))
        f.write("%d step, %s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f\n" % (i+1, name, mode, MSE, RMSE, MAE, MAPE))
    f.close()
    print('Model Testing Ended ...', time.ctime())


P = type('Parameters', (object,), {})()
P.DATANAME = 'METRLA'
P.MODEL = 'LSTM'
P.seed = 0
P.T_TRN = 0.7
P.T_VAL = 0.1
P.S_TRN = 0.7
P.S_VAL = 0.1
P.TIMESTEP_IN = 12
P.TIMESTEP_OUT = 12
P.CHANNEL = 1
P.BATCHSIZE = 64
P.hidden_dim = 128
P.TEMPERATURE = 1
P.LEARN = 0.001
P.PRETRN_EPOCH = 5
P.EPOCH = 2
P.weight_decay = 0
P.IS_PRETRN = True
P.is_adp_adj = True
P.is_SGA = False
P.is_GCN = True
P.is_sampler = False
# not possible: gcn false and sampler false
P.is_testunseen = True

if torch.backends.mps.is_available():
    device = torch.device('mps') 
if torch.cuda.is_available():
    device = torch.device('cuda')

def main():
    P.KEYWORD = 'pred_' + P.DATANAME + '_' + '_' + datetime.now().strftime("%y%m%d%H%M") + '_' + str(os.getpid())
    P.PATH = 'save/' + P.KEYWORD
    global data
    if P.DATANAME == 'METRLA':
        print('P.DATANAME == METRLA')
        P.FLOWPATH = './data/METRLA/metr-la.h5'
        P.n_dct_coeff = 3918
        P.ADJPATH = './data/METRLA/adj_mx.pkl'
        P.N_NODE = 207
        data = pd.read_hdf(P.FLOWPATH).values
    print('data.shape:', data.shape)
    pretrn_iter, preval_iter, spatialSplit_unseen, spatialSplit_allNod, \
        train_iter, val_u_iter, val_a_iter, tst_u_iter, tst_a_iter, \
        adj_train, adj_val_u, adj_val_a, adj_tst_u, adj_tst_a = setups()
    # Now, only use pretrn_iter for encoding
    save_parameters(P, P.PATH + '/' + 'parameters.txt')
    if P.IS_PRETRN:
        pretrainModel('encoder', pretrn_iter, preval_iter, adj_train, adj_val_a, device, spatialSplit_allNod)
    # print(edge_masking(adj_train, 0.9, device))
    trainModel(P.MODEL, 'train',
        train_iter, val_u_iter, val_a_iter,
        adj_train, adj_val_u, adj_val_a,
        spatialSplit_unseen, spatialSplit_allNod)
    testModel(P.MODEL, 'test_a', tst_a_iter, adj_tst_a, spatialSplit_allNod)


if __name__ == '__main__':
    main()

    