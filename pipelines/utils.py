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

def get_binary_metrics(preds, labels, bootstrap=False):
    accuracy = Accuracy(task="binary", threshold=threshold).to(labels.device)
    auroc = AUROC(task="binary").to(labels.device)
    auprc = AveragePrecision(task="binary").to(labels.device)
    f1 = BinaryF1Score().to(labels.device)
    # convert labels type to int
    labels = labels.type(torch.int)

    if not bootstrap:
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
    else:
        accuracy_list = []
        auroc_list = []
        auprc_list = []
        minpse_list = []
        f1_list = []
        for _ in range(10):
            # 生成相同的随机索引
            n = int(0.9 * preds.size(0))
            random_indices = torch.randperm(preds.size(0))[:n]

            # 根据随机的索引选择行
            sampled_preds = preds[random_indices]
            sampled_labels = labels[random_indices]

            accuracy(sampled_preds, sampled_labels)
            auroc(sampled_preds, sampled_labels)
            auprc(sampled_preds, sampled_labels)
            f1(sampled_preds, sampled_labels)
            minpse_score = minpse(sampled_preds, sampled_labels) 

            accuracy_list.append(accuracy.compute().item())
            auroc_list.append(auroc.compute().item())
            auprc_list.append(auprc.compute().item())
            minpse_list.append(minpse_score)
            f1_list.append(f1.compute().item())
        return {
            "accuracy": np.mean(accuracy_list),
            "auroc": np.mean(auroc_list),
            "auprc": np.mean(auprc_list),
            "minpse": np.mean(minpse_list),
            "f1": np.mean(f1_list),
            "accuracy-std": np.std(accuracy_list),
            "auroc-std": np.std(auroc_list),
            "auprc-std": np.std(auprc_list),
            "minpse-std": np.std(minpse_list),
            "f1-std": np.std(f1_list)
        }



def set_seed(RANDOM_SEED):
    np.random.seed(RANDOM_SEED) #numpy
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED) # cpu
    torch.cuda.manual_seed(RANDOM_SEED) #gpu
    torch.backends.cudnn.deterministic=True # cudnn