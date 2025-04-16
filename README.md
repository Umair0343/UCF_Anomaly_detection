# Video Anomaly Detection
CLAWS: Clustering Assisted Weakly Supervised Learning with Normalcy Suppression for Anomalous Event Detection
Paper link: https://arxiv.org/abs/2011.12077
This repository contains an implementation of a deep learning model for anomaly detection in videos using the UCF Crime dataset, for the research paper mentioned above. The model leverages clustering techniques and a Multi-Layer Perceptron (MLP) architecture to identify anomalous events in video segments.

## Features

- Anomaly detection in surveillance videos
- Feature-based clustering for better anomaly representation
- Training with normal and anomalous video segments
- Evaluation using AUC (Area Under the ROC Curve)

## Model Architecture

The model consists of the following components:

1. **BBN (Backbone Network)**: A neural network with two attention modules (NSB1 and NSB2) that handle feature weighting
2. **Clustering Module**: Utilizes K-means clustering to separate normal and anomalous patterns
3. **Loss Function**: Combines regression loss, temporal smoothness, sparsity, and clustering-based loss

## Project Structure

```
anomaly-detection/
├── data/                  # Data directory (not included in repository)
├── models/                # Model architecture definitions
├── datasets/              # Dataset handling code
├── utils/                 # Utility functions for metrics, visualization, etc.
├── train.py               # Training script
├── test.py                # Testing script
├── config.py              # Configuration parameters
├── requirements.txt       # Dependencies
└── README.md              # This file
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Umair0343/UCF_Anomaly_detection.git
cd UCF_Anomaly_detection
```

2. Install dependencies:
```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
pip install -r requirements.txt
```

3. Prepare the data:
   - Download the UCF-Crime Dataset: https://www.crcv.ucf.edu/chenchen/dataset.html
   - Extract I3D features. You can also download extracted i3d features from here: https://drive.google.com/drive/folders/1PBgv7xwdxU9RQI9ePuWG1jTLBaKs0ETG?usp=drive_link
   - Organize them according to the expected directory structure

## Usage

### Training

To train the model with default parameters:

```bash
python train.py
```

With custom parameters:

```bash
python train.py --data_dir /path/to/training/data --test_dir /path/to/test/data --epochs 10 --lr 0.0005
```

### Testing

To evaluate a trained model:

```bash
python test.py --model_path /path/to/model.pth --results_dir results
```

## Results

The model's performance is evaluated using the AUC metric. The ROC curve and other visualizations are saved to the specified results directory after testing.

## Citation

If you use this code in your research, please cite:

```
@article{anomaly_detection,
  title={Video Anomaly Detection with Clustering-based Feature Learning},
  author={Your Name},
  journal={arXiv preprint},
  year={2023}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
