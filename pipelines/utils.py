import numpy as np
import random

import torch
from torchmetrics import AUROC, Accuracy, AveragePrecision, Precision, Recall, F1Score
from torchmetrics.classification import BinaryF1Score, ConfusionMatrix
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError, R2Score
from sklearn import metrics as sklearn_metrics


def minpse(preds, labels):
    precisions, recalls, thresholds = sklearn_metrics.precision_recall_curve(labels.cpu().detach().numpy(), preds.cpu().detach().numpy())
    minpse_score = np.max([min(x, y) for (x, y) in zip(precisions, recalls)])
    return minpse_score

# get regression metrics: mse, mae, rmse, r2

def get_regression_metrics(preds, labels):
    mse = MeanSquaredError(squared=True).to(labels.device)
    rmse = MeanSquaredError(squared=False).to(labels.device)
    mae = MeanAbsoluteError().to(labels.device)
    r2 = R2Score().to(labels.device)

    mse(preds, labels)
    rmse(preds, labels)
    mae(preds, labels)
    r2(preds, labels)

    # return a dictionary
    return {
        "mse": mse.compute().item(),
        "rmse": rmse.compute().item(),
        "mae": mae.compute().item(),
        "r2": r2.compute().item(),
    }

threshold = 0.5

def get_binary_metrics(preds, labels):
    accuracy = Accuracy(task="binary", threshold=threshold).to(labels.device)
    auroc = AUROC(task="binary").to(labels.device)
    auprc = AveragePrecision(task="binary").to(labels.device)
    f1 = BinaryF1Score().to(labels.device)
    # convert labels type to int
    labels = labels.type(torch.int)

    accuracy(preds, labels)
    auroc(preds, labels)
    auprc(preds, labels)
    f1(preds, labels)
    minpse_score = minpse(preds, labels) 

    # return a dictionary
    return {
        "accuracy": accuracy.compute().item(),
        "auroc": auroc.compute().item(),
        "auprc": auprc.compute().item(),
        "minpse": minpse_score,
        "f1": f1.compute().item(),
    }



def set_seed(RANDOM_SEED):
    np.random.seed(RANDOM_SEED) #numpy
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED) # cpu
    torch.cuda.manual_seed(RANDOM_SEED) #gpu
    torch.backends.cudnn.deterministic=True # cudnn