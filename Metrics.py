import numpy as np
import torch

def evaluate(y_true, y_pred):
    return MSE(y_true, y_pred), RMSE(y_true, y_pred), MAE(y_true, y_pred), MAPE(y_true, y_pred)

def evaluate_all(y_true, y_pred):
    return MSE_all(y_true, y_pred), RMSE_all(y_true, y_pred), MAE_all(y_true, y_pred), MAPE_all(y_true, y_pred)

# def MSE(y_true, y_pred):
#     # y_true[y_true < 1e-5] = 0
#     # y_pred[y_pred < 1e-5] = 0
#     with np.errstate(divide = 'ignore', invalid = 'ignore'):
#         mask = np.not_equal(y_true, 0)
#         mask = mask.astype(np.float32)
#         mask /= np.mean(mask)
#         mse = np.square(y_pred - y_true)
#         mse = np.nan_to_num(mse * mask)
#         mse = np.mean(mse)
#         return mse

# def MSE_all(y_true, y_pred):
#     with np.errstate(divide = 'ignore', invalid = 'ignore'):
#         mse = np.square(y_pred - y_true)
#         mse = np.mean(mse)
#         return mse
    
# def RMSE(y_true, y_pred):
#     # y_true[y_true < 1e-5] = 0
#     # y_pred[y_pred < 1e-5] = 0
#     with np.errstate(divide = 'ignore', invalid = 'ignore'):
#         mask = np.not_equal(y_true, 0)
#         mask = mask.astype(np.float32)
#         mask /= np.mean(mask)
#         rmse = np.square(np.abs(y_pred - y_true))
#         rmse = np.nan_to_num(rmse * mask)
#         rmse = np.sqrt(np.mean(rmse))
#         return rmse

# def RMSE_all(y_true, y_pred):
#     with np.errstate(divide = 'ignore', invalid = 'ignore'):
#         rmse = np.square(np.abs(y_pred - y_true))
#         rmse = np.sqrt(np.mean(rmse))
#         return rmse
        
# def MAE(y_true, y_pred):
#     # y_true[y_true < 1e-5] = 0
#     # y_pred[y_pred < 1e-5] = 0
#     with np.errstate(divide = 'ignore', invalid = 'ignore'):
#         mask = np.not_equal(y_true, 0)
#         mask = mask.astype(np.float32)
#         mask /= np.mean(mask)
#         mae = np.abs(y_pred - y_true)
#         mae = np.nan_to_num(mae * mask)
#         mae = np.mean(mae)
#         return mae
    
# def MAE_all(y_true, y_pred):
#     with np.errstate(divide = 'ignore', invalid = 'ignore'):
#         mae = np.abs(y_pred - y_true)
#         mae = np.mean(mae)
#         return mae

# def MAPE(y_true, y_pred):
#     # y_true[y_true < 1e-5] = 0
#     # y_pred[y_pred < 1e-5] = 0
#     with np.errstate(divide='ignore', invalid='ignore'):
#         mask = np.not_equal(y_true, 0)
#         mask = mask.astype('float32')
#         mask /= np.mean(mask)
#         mape = np.abs(np.divide((y_pred - y_true).astype('float32'), y_true))
#         mape = np.nan_to_num(mask * mape)
#         return np.mean(mape) * 100

# def MAPE_all(y_true, y_pred):
#     with np.errstate(divide='ignore', invalid='ignore'):
#         mape = np.abs(np.divide((y_pred - y_true).astype('float32'), y_true))
#         return np.mean(mape) * 100
    
def MSE(y_true, y_pred):
    mask = y_true != 0
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    mse = torch.square(y_pred - y_true)
    mse = mse * mask
    mse = torch.where(torch.isnan(mse), torch.zeros_like(mse), mse)
    return torch.mean(mse)

def MSE_all(y_true, y_pred):
    mse = torch.square(y_pred - y_true)
    mse = torch.mean(mse)
    return mse

def RMSE(y_true, y_pred):
    mask = y_true != 0
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    rmse = torch.square(torch.abs(y_pred - y_true))
    rmse = rmse * mask
    rmse = torch.where(torch.isnan(rmse), torch.zeros_like(rmse), rmse)
    return torch.sqrt(torch.mean(rmse))

def RMSE_all(y_true, y_pred):
    rmse = torch.square(torch.abs(y_pred - y_true))
    rmse = torch.mean(rmse)
    return torch.sqrt(rmse)

def MAE(y_true, y_pred):
    mask = y_true != 0
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    mae = torch.abs(y_pred - y_true)
    mae = mae * mask
    mae = torch.where(torch.isnan(mae), torch.zeros_like(mae), mae)
    return torch.mean(mae)

def MAE_all(y_true, y_pred):
    mae = torch.abs(y_pred - y_true)
    return torch.mean(mae)

def MAPE(y_true, y_pred):
    mask = y_true != 0
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    mape = torch.abs(torch.divide((y_pred - y_true).float(), y_true))
    mape = mape * mask
    mape = torch.where(torch.isnan(mape), torch.zeros_like(mape), mape)
    return torch.mean(mape) * 100

def MAPE_all(y_true, y_pred):
    mape = torch.abs(torch.divide((y_pred - y_true).float(), y_true))
    return torch.mean(mape) * 100