import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle

def plot_loss_curve(losses, title="Training Loss Curve", save_path=None):
    """
    Plot loss curve over epochs
    """
    plt.figure(figsize=(10, 6))
    plt.plot(losses, 'b-', label='Loss')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_auc_curve(all_auc, title="AUC over Epochs", save_path=None):
    """
    Plot AUC curve over epochs
    """
    plt.figure(figsize=(10, 6))
    plt.plot(all_auc, 'r-', label='AUC')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.grid(True)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_roc_curve(fpr, tpr, roc_auc, title="ROC Curve", save_path=None):
    """
    Plot ROC curve
    """
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    plt.show()
