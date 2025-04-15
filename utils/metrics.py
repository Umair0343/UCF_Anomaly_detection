import numpy as np
import torch
from sklearn.metrics import roc_curve, auc
from prettytable import PrettyTable

def count_parameters(model):
    """
    Count and display trainable parameters in the model
    """
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

def auc_calculator(true_labels, predicted_scores):
    """
    Calculate AUC from true labels and predicted scores
    """
    true_labels_flatten = [item.cpu().numpy() for sublist in true_labels for item in sublist]
    predicted_scores_flatten = [item.cpu().numpy() for sublist in predicted_scores for item in sublist]

    true = np.array(true_labels_flatten)
    pre = np.array(predicted_scores_flatten)
    fpr, tpr, threshold = roc_curve(true_labels_flatten, predicted_scores_flatten)
    auc_res = auc(fpr, tpr)

    return auc_res, fpr, tpr