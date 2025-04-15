import torch
import os

# Set default tensor type to CUDA if available
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

# Paths
DATA_DIR_TRAIN = '/home/iml1/Desktop/UMAIR/ucf/i3d_features/training'
DATA_DIR_TEST = '/home/iml1/Desktop/UMAIR/ucf/i3d_features/test'
ANNOTATION_FILE = '/home/iml1/Desktop/UMAIR/ucf/i3d_features/Temporal_Anomaly_Annotation_for_Testing_Videos.txt'
MODEL_SAVE_PATH = '/home/iml1/Desktop/UMAIR/ucf/UCF_Models'

# Model parameters
INPUT_SIZE = 1024
HIDDEN_SIZE = 512
OUTPUT_SIZE = 32
DROPOUT_RATE = 0.6

# Training parameters
BATCH_SIZE = 1
EPOCHS = 10
LEARNING_RATE = 0.001
MILESTONES = [80000]
LR_GAMMA = 0.1

# Clustering parameters
N_CLUSTERS = 2
ALPHA = 0.7
BETA = 0.25

# Loss parameters
LAMBDA1 = 0.00008
LAMBDA2 = 0.00008

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')