import torch
import numpy as np
import pandas as pd
import sys
import os
import time
from datetime import datetime
import unseen_nodes
from utils import load_pickle
from encoder import *

def getXSYS(data, mode):
    TRAIN_NUM = int(data.shape[0] * P.TRAINRATIO)
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

def load_adj(pkl_filename, adjtpe, dataname):
    if dataname == 'METRLA' or dataname == 'PEMSBAY':
        sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(pkl_filename)
    adj =  [adj_mx]
    return adj

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
    P.train_size = int(P.trainval_size * (1-P.TRAINVALSPLIT))
    XS_torch_trn = trainXS[:P.train_size,:,:,:]
    YS_torch_trn = trainYS[:P.train_size,:,:,:]
    XS_torch_val = trainXS[P.train_size:P.trainval_size,:,:,:]
    YS_torch_val = trainYS[P.train_size:P.trainval_size,:,:,:]
    print('XS_torch_trn.shape', XS_torch_trn.shape)
    print('YS_torch_trn.shape', YS_torch_trn.shape)
    print('XS_torch_val.shape', XS_torch_val.shape)
    print('YS_torch_val.shape', YS_torch_val.shape)
    # # spatial split
    spatialSplit_unseen = unseen_nodes.SpatialSplit(data.shape[1], r_trn=P.R_TRN, r_val=.1, r_tst=.2, seed=P.seed)
    print('spatialSplit_unseen', spatialSplit_unseen)
    # print(spatialSplit_unseen.i_trn)
    # print(spatialSplit_unseen.i_val)
    # print(spatialSplit_unseen.i_tst)
    XS_torch_train = torch.Tensor(XS_torch_trn[:,:,spatialSplit_unseen.i_trn,:])
    YS_torch_train = torch.Tensor(YS_torch_trn[:,:,spatialSplit_unseen.i_trn,:])
    XS_torch_val_u = torch.Tensor(XS_torch_val[:,:,spatialSplit_unseen.i_val,:])
    YS_torch_val_u = torch.Tensor(YS_torch_val[:,:,spatialSplit_unseen.i_val,:])
    XS_torch_tst_u = torch.Tensor(testXS[:,:,spatialSplit_unseen.i_tst,:])
    YS_torch_tst_u = torch.Tensor(testYS[:,:,spatialSplit_unseen.i_tst,:])
    print('train.shape', XS_torch_train.shape, YS_torch_train.shape)
    print('val_u.shape', XS_torch_val_u.shape, YS_torch_val_u.shape)
    print('tst_u.shape', XS_torch_tst_u.shape, YS_torch_tst_u.shape)
    # torch dataset
    train_data = torch.utils.data.TensorDataset(XS_torch_train, YS_torch_train)
    val_u_data = torch.utils.data.TensorDataset(XS_torch_val_u, YS_torch_val_u)
    tst_u_data = torch.utils.data.TensorDataset(XS_torch_tst_u, YS_torch_tst_u)
    # torch dataloader
    train_iter = torch.utils.data.DataLoader(train_data, P.BATCHSIZE, shuffle=True)
    val_u_iter = torch.utils.data.DataLoader(val_u_data, P.BATCHSIZE, shuffle=False)
    tst_u_iter = torch.utils.data.DataLoader(tst_u_data, P.BATCHSIZE, shuffle=False)
    # adj matrix spatial split
    adj_mx = load_adj(P.ADJPATH, None, P.DATANAME)
    adj_train = [torch.tensor(i[spatialSplit_unseen.i_trn,:][:,spatialSplit_unseen.i_trn]).to(device) for i in adj_mx]
    adj_val_u = [torch.tensor(i[spatialSplit_unseen.i_val,:][:,spatialSplit_unseen.i_val]).to(device) for i in adj_mx]
    adj_tst_u = [torch.tensor(i[spatialSplit_unseen.i_tst,:][:,spatialSplit_unseen.i_tst]).to(device) for i in adj_mx]
    print('adj_train', len(adj_train), adj_train[0].shape)
    # print('adj_train', len(adj_train), adj_train[0].shape, adj_train[1].shape)
    # print('adj_val_u', len(adj_val_u), adj_val_u[0].shape, adj_val_u[1].shape)
    # print('adj_tst_u', len(adj_tst_u), adj_tst_u[0].shape, adj_tst_u[1].shape)
    # PRETRAIN data loader
    pretrn_iter = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(
        XS_torch_train[:,-1,:,0].T), batch_size=1, shuffle=True)
    preval_iter = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(
        torch.tensor(trainYS[:,-1,spatialSplit_unseen.i_val,0]).T.float()),
    batch_size=1, shuffle=False)
    print('pretrn_iter.dataset.tensors[0].shape', pretrn_iter.dataset.tensors[0].shape)
    print('preval_iter.dataset.tensors[0].shape', preval_iter.dataset.tensors[0].shape)
    # print
    # for k, v in vars(P).items():
    #     print(k,v)
    return pretrn_iter, preval_iter, spatialSplit_unseen, \
        train_iter, val_u_iter, tst_u_iter, \
        adj_train, adj_val_u, adj_tst_u

def pretrainModel(pretrain_iter, preval_iter, adj):
    print('pretrainModel Started ...', time.ctime())
    model = Contrastive_FeatureExtractor_conv(P.TEMPERATURE, adj).to(device)
    min_val_loss = np.inf
    optimizer = torch.optim.Adam(model.parameters(), lr=P.LEARN, weight_decay=P.weight_decay)
    loss_sum, n = 0.0, 0
    model.train()
    x = pretrain_iter.dataset.tensors
    print('x[0].shape', x[0].shape)
    optimizer.zero_grad()
    loss = model.contrast(x[0].to(device))
    loss.backward()
    optimizer.step()
    loss_sum += loss.item() * x[0].shape[0]
    n += x[0].shape[0]
    train_loss = loss_sum / n
    # val_loss = pre_evaluateModel(model, preval_iter)
    # if val_loss < min_val_loss:
    #     min_val_loss = val_loss
    #     torch.save(model.state_dict(), P.PATH + '/' + name + '.pt')
    # endtime = datetime.now()
    # epoch_time = (endtime - starttime).seconds
    # print("epoch", epoch, "time used:", epoch_time," seconds ", "train loss:", train_loss, "validation loss:", val_loss)
    # with open(P.PATH + '/' + name + '_log.txt', 'a') as f:
    #     f.write("%s, %d, %s, %d, %s, %s, %.10f, %s, %.10f\n" % ("epoch", epoch, "time used", epoch_time, "seconds", "train loss", train_loss, "validation loss:", val_loss))

P = type('Parameters', (object,), {})()
P.DATANAME = 'METRLA'
P.seed = 0
P.TRAINRATIO = 0.8 # TRAIN + VAL
P.TIMESTEP_IN = 12
P.TIMESTEP_OUT = 12
P.CHANNEL = 1
P.TRAINVALSPLIT = 0.125 # val_ratio = 0.8 * 0.125 = 0.1
P.R_TRN = 0.7
P.BATCHSIZE = 64
P.TEMPERATURE = 1
P.LEARN = 0.001
P.weight_decay = 0

device = torch.device('mps') 

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
    pretrn_iter, preval_iter, spatialSplit_unseen, \
        train_iter, val_u_iter, tst_u_iter, \
        adj_train, adj_val_u, adj_tst_u = setups()
    # Now, only use pretrn_iter for encoding
    pretrainModel(pretrn_iter, preval_iter, adj_train)


if __name__ == '__main__':
    main()

    