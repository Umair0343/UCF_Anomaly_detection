import os
import time
import random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
import argparse

from datasets import UCFDataset
from models import BBN, Clustering
from utils import LossCalculator, count_parameters, auc_calculator
from utils import plot_loss_curve, plot_auc_curve
import config

def setup_logging():
    """Set up logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('training.log')
        ]
    )
    return logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train anomaly detection model on UCF Crime dataset')
    parser.add_argument('--data_dir', type=str, default=config.DATA_DIR_TRAIN,
                        help='Directory of the training data')
    parser.add_argument('--test_dir', type=str, default=config.DATA_DIR_TEST,
                        help='Directory of the test data')
    parser.add_argument('--annotation_file', type=str, default=config.ANNOTATION_FILE,
                        help='Path to annotation file')
    parser.add_argument('--model_save_path', type=str, default=config.MODEL_SAVE_PATH,
                        help='Directory to save model checkpoints')
    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=config.EPOCHS,
                        help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=config.LEARNING_RATE,
                        help='Learning rate')
    
    return parser.parse_args()

def main():
    """Main training function"""
    args = parse_args()
    logger = setup_logging()
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    random.seed(42)
    
    # Ensure model save directory exists
    os.makedirs(args.model_save_path, exist_ok=True)
    
    logger.info(f"Using device: {config.DEVICE}")

    # Load datasets
    logger.info("Loading datasets...")
    train_dataset = UCFDataset(args.data_dir)
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        generator=torch.Generator(device='cuda' if torch.cuda.is_available() else 'cpu')
    )

    test_dataset = UCFDataset(
        args.test_dir, 
        test=True, 
        annotation_file=args.annotation_file
    )
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        generator=torch.Generator(device='cuda' if torch.cuda.is_available() else 'cpu')
    )

    # Initialize model
    logger.info("Initializing model...")
    model = BBN(
        config.INPUT_SIZE, 
        config.HIDDEN_SIZE, 
        config.OUTPUT_SIZE, 
        dropout_rate=config.DROPOUT_RATE
    )
    model.to(config.DEVICE)
    logger.info(str(model))
    count_parameters(model)

    # Optimizer and scheduler
    optimizer = optim.RMSprop(model.parameters(), lr=args.lr)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, 
        milestones=config.MILESTONES, 
        gamma=config.LR_GAMMA
    )

    # Training metrics
    best_auc = 0.0
    losses_curve = []
    all_auc = []

    # Training loop
    logger.info("Starting training...")
    for epoch in range(args.epochs):
        logger.info(f"\nEpoch {epoch+1}/{args.epochs} -----------------------------")
        start_time = time.time()
        model.train()
        all_losses = []

        progress_bar = tqdm(
            enumerate(train_dataloader), 
            total=len(train_dataloader), 
            desc="Training", 
            ncols=100
        )

        for batch_idx, (segments, labels) in progress_bar:
            torch_labels = torch.stack(labels).squeeze(1)
            train_vid_label = torch.sum(torch_labels)
            clus_lbl = 1 if train_vid_label > 1 else 0

            torch_segments = torch.stack(segments).squeeze(1).view(-1, config.INPUT_SIZE).to(config.DEVICE)
            indices = random.sample(range(len(segments)), len(segments))

            for index in indices:
                seg_output = model.fc1(torch_segments)
                clustering = Clustering(
                    n_clusters=config.N_CLUSTERS,
                    alpha=config.ALPHA,
                    beta=config.BETA
                )
                cluster_loss = clustering.clusters(seg_output, clus_lbl)

                train_segments = segments[index].to(config.DEVICE)
                train_labels = labels[index].to(config.DEVICE)
                pred_y = model(train_segments.squeeze(0)).squeeze(0)
                train_labels = train_labels.T

                loss_calculator = LossCalculator(
                    pred_y, 
                    train_labels, 
                    cluster_loss,
                    lambda1=config.LAMBDA1,
                    lambda2=config.LAMBDA2
                )
                losses = loss_calculator.total_loss(pred_y, train_labels)

                all_losses.append(losses.detach().cpu().numpy())

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()
                lr_scheduler.step()

                # Update tqdm status bar with loss
                progress_bar.set_postfix({"loss": f"{losses.item():.4f}"})

        mean_loss = np.mean(all_losses)
        losses_curve.append(mean_loss)
        logger.info(f"  Finished training epoch {epoch+1}, Avg loss: {mean_loss:.4f}")

        torch.cuda.empty_cache()

        # Evaluation
        logger.info("  Evaluating model on test set...")
        model.eval()
        true_labels = []
        predicted_scores = []

        with torch.no_grad():
            for idx, (segments, labels) in enumerate(test_dataset):
                segment = segments.to(config.DEVICE)
                te_label = labels.to(config.DEVICE)
                te_pred_y = model(segment)

                t_label = te_label.reshape(len(te_label), 1)
                t_pred_y = te_pred_y.reshape(len(te_pred_y), 1)

                true_labels.append(t_label)
                predicted_scores.append(t_pred_y)

                del segment, t_label, t_pred_y
                torch.cuda.empty_cache()

        auc_res, _, _ = auc_calculator(true_labels, predicted_scores)
        all_auc.append(auc_res)
        logger.info(f"  Test AUC: {auc_res:.4f}")

        if auc_res > best_auc:
            logger.info("  New best model found. Saving...")
            best_auc = auc_res
            torch.save(model.state_dict(), os.path.join(args.model_save_path, 'best_model.pth'))

        if (epoch + 1) % 5 == 0 or epoch == 0:
            torch.save(model.state_dict(), os.path.join(args.model_save_path, f'epoch_{epoch+1}.pth'))

        end_time = time.time()
        logger.info(f"  Epoch {epoch+1} completed in {(end_time - start_time)/60:.2f} minutes")

    logger.info("\nTraining completed!")
    logger.info(f"Best AUC achieved: {best_auc:.4f}")
    
    # Plot training curves
    plot_loss_curve(losses_curve, save_path=os.path.join(args.model_save_path, 'loss_curve.png'))
    plot_auc_curve(all_auc, save_path=os.path.join(args.model_save_path, 'auc_curve.png'))


if __name__ == "__main__":
    main()