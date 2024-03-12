import pickle
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

def load_adj(pkl_filename, adjtpe, dataname):
    if dataname == 'PEMSBAY':
        sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(pkl_filename)
        adj =  [adj_mx]
    elif dataname == 'METRLA':
        sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(pkl_filename)
        adj = []
        adj.append(adj_mx)
        adj.append(adj_mx.T)
    return adj