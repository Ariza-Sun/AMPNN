import torch
import torchmetrics
import numpy as np
import pandas as pd
import os
import math
from sklearn.metrics import roc_curve, auc

# if use cuda
use_cuda = torch.cuda.is_available()

# eval on dyn
def accuracy(pred, y):
    """
    :param pred: predictions
    :param y: ground truth
    :return: accuracy, defined as 1 - (norm(y - pred) / norm(y))
    """
    if torch.linalg.norm(y, "fro")==0:
        return torch.tensor(1)
    else:
        return 1 - torch.linalg.norm(y - pred, "fro") / torch.linalg.norm(y, "fro") # use Frobenius norm


def r2(pred, y):
    """
    :param y: ground truth
    :param pred: predictions
    :return: R square (coefficient of determination)
    """
    if torch.sum((y - torch.mean(y)) ** 2)==0:
        return torch.tensor(1)
    else:
        return 1 - torch.sum((y - pred) ** 2) / torch.sum((y - torch.mean(y)) ** 2)


def explained_variance(pred, y):
    return 1 - torch.var(y - pred) / torch.var(y)


def cal_dyn_metrics(predictions,y):
   rmse = torch.sqrt(torchmetrics.functional.mean_squared_error(predictions, y))
   mae = torchmetrics.functional.mean_absolute_error(predictions, y)
   acc =accuracy(predictions, y)
   r_2 =r2(predictions, y)
   explainedvariance = explained_variance(predictions, y)
   return rmse, mae, acc, r_2, explainedvariance

def dyn_evaluator(predictions,y,batch_size,max_num,min_num):
    '''
    evaluate for a batch, y.shape=(batchsize,nodesize, dim)
    '''
    # squeeze pre_len=1
    predictions=torch.squeeze(predictions, dim=1)
    y=torch.squeeze(y, dim=1)

    predictions=predictions*(max_num-min_num)+min_num
    y=y*(max_num-min_num)+min_num

    
    #print(predictions.shape,y.shape) (bs, node_num)
    rmse, mae, accuracy, r2, explained_variance=cal_dyn_metrics(predictions,y)
    return rmse.item(), mae.item(), accuracy.item(), r2.item(), explained_variance.item()