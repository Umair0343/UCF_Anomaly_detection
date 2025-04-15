import os
import torch
import argparse
import numpy as np
from tqdm import tqdm
import logging

from datasets import UCFDataset
from models import BBN
from utils import auc_calculator, plot_roc_curve
import config

def setup_logging():
    """Set up logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('testing.log')
        ]
    )
    return logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Test anomaly detection model on UCF Crime dataset')
    parser.add_argument('--test_dir', type=str, default=config.DATA_DIR_TEST,
                        help='Directory of the test data')
    parser.add_argument('--annotation_file', type=str, default=config.ANNOTATION_FILE,
                        help='Path to annotation file')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the model checkpoint to test')
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Directory to save test results')
    
    return parser.parse_args()

def main():
    """Main testing function"""
    args = parse_args()
    logger = setup_logging()
    
    # Ensure results directory exists
    os.makedirs(args.results_dir, exist_ok=True)
    
    logger.info(f"Using device: {config.DEVICE}")

    # Load test dataset
    logger.info("Loading test dataset...")
    test_dataset = UCFDataset(
        args.test_dir, 
        test=True, 
        annotation_file=args.annotation_file
    )
    
    # Load model
    logger.info(f"Loading model from {args.model_path}...")
    model = BBN(
        config.INPUT_SIZE, 
        config.HIDDEN_SIZE, 
        config.OUTPUT_SIZE, 
        dropout_rate=config.DROPOUT_RATE
    )
    model.to(config.DEVICE)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()
    
    # Testing
    logger.info("Starting evaluation...")
    true_labels = []
    predicted_scores = []
    
    with torch.no_grad():
        for idx, (segments, labels) in enumerate(tqdm(test_dataset, desc="Testing")):
            segment = segments.to(config.DEVICE)
            label = labels.to(config.DEVICE)
            pred_y = model(segment)
            
            t_label = label.reshape(len(label), 1)
            t_pred_y = pred_y.reshape(len(pred_y), 1)
            
            true_labels.append(t_label)
            predicted_scores.append(t_pred_y)
            
            del segment, t_label, t_pred_y
            torch.cuda.empty_cache()
    
    # Calculate metrics
    auc_res, fpr, tpr = auc_calculator(true_labels, predicted_scores)
    logger.info(f"Test AUC: {auc_res:.4f}")
    
    # Plot ROC curve
    plot_roc_curve(
        fpr, 
        tpr, 
        auc_res, 
        title="ROC Curve for Anomaly Detection", 
        save_path=os.path.join(args.results_dir, 'roc_curve.png')
    )
    
    # Save results
    results = {
        'auc': auc_res,
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist()
    }
    
    np.save(os.path.join(args.results_dir, 'test_results.npy'), results)
    logger.info(f"Results saved to {args.results_dir}")

if __name__ == "__main__":
    main()