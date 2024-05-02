import pickle
import numpy as np
import scipy.sparse as sp

def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data

def load_adj(pkl_filename, adjtype, dataname, diag):
    sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(pkl_filename)
    
    if adjtype == "transition":
        adj1 = asym_adj(adj_mx)
        np.fill_diagonal(adj1, diag)
        adj = [adj1]
    elif adjtype == "doubletransition":
        adj1 = asym_adj(adj_mx)
        adj2 = asym_adj(adj_mx.T)
        np.fill_diagonal(adj1, diag)
        np.fill_diagonal(adj2, diag)
        adj = [adj1, adj2]
    elif adjtype == "original":
        adj1 = adj_mx
        np.fill_diagonal(adj1, diag)
        adj = [adj1]
    elif adjtype == "doubleoriginal":
        adj1 = adj_mx
        adj2 = adj_mx.T
        np.fill_diagonal(adj1, diag)
        np.fill_diagonal(adj2, diag)
        adj = [adj1, adj2]
    elif adjtype == "identity":
        adj1 = np.eye(len(sensor_ids))
        np.fill_diagonal(adj1, diag)
        adj = [adj1]
    return adj

def asym_adj(adj):
    np.fill_diagonal(adj, 0)
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten().astype(np.float64)  # Ensure floating-point calculations
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv)
    output = np.array(d_mat.dot(adj).astype(np.float32).todense())
    np.fill_diagonal(output, 1)
    return output