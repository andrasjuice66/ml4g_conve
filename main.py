import torch
from pathlib import Path
import argparse
from models.conve import ConvE
from utils.preprocess import KGDataLoader, create_dataloader
from utils.train import train_conve, evaluate
import wandb
import os
import datetime

def main(data_path, dataset='FB15k-237'):
    wandb.finish()
    # Initialize wandb
    wandb_api_key = 'a15aa5a84ab821022d13f9aa3a59ec1770fe93a3'
    wandb.login(key=wandb_api_key)

    run_name = f"conve_{dataset}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    wandb.init(
        project="conve-kg",
        name=run_name,
        config={  # Log hyperparameters here
            "dataset": dataset,
            "embedding_dim": 200,
            "embedding_shape1": 20,
            "input_dropout": 0.2,
            "hidden_dropout": 0.3,
            "feature_map_dropout": 0.2,
            "num_epochs": 1000,
            "batch_size": 128,
            "learning_rate": 0.003,
            "label_smoothing": 0.1
        }
    )
    config = wandb.config

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load datasets
    print("Loading data...")
    data_loader = KGDataLoader(f"{data_path}{dataset}")
    datasets = data_loader.load_data()

    # Create dataloaders
    train_loader = create_dataloader(datasets['train'], batch_size=config.batch_size, shuffle=True)
    valid_loader = create_dataloader(datasets['valid'], batch_size=config.batch_size, shuffle=False)
    test_loader = create_dataloader(datasets['test'], batch_size=config.batch_size, shuffle=False)

    # Initialize model
    print("Initializing model...")
    model = ConvE(
        num_entities=len(data_loader.entity2id),
        num_relations=len(data_loader.relation2id),
        embedding_dim=config.embedding_dim,
        embedding_shape1=config.embedding_shape1,
        input_dropout=config.input_dropout,
        hidden_dropout=config.hidden_dropout,
        feature_map_dropout=config.feature_map_dropout,
    ).to(device)

    # Training parameters
    train_params = {
        'num_epochs': config.num_epochs,
        'learning_rate': config.learning_rate,
        'label_smoothing': config.label_smoothing,
        'save_path': "checkpoints",
        'eval_every': 1  # Evaluate every epoch
    }

    # Train model
    print("Starting training...")
    model = train_conve(
        model=model,
        train_dataloader=train_loader,
        valid_dataloader=valid_loader,
        device=device,
        **train_params  
    )

    # Final evaluation on test set
    print("\nEvaluating on test set...")
    test_metrics = evaluate(model, test_loader, device)

    # Log final test metrics
    wandb.log({
        'test_mr': test_metrics['mr'],
        'test_mrr': test_metrics['mrr'],
        'test_hits@1': test_metrics['hits@1'],
        'test_hits@3': test_metrics['hits@3'],
        'test_hits@10': test_metrics['hits@10'],
    })

    print("\nFinal Test Metrics:")
    print(f"MR: {test_metrics['mr']:.1f}")
    print(f"MRR: {test_metrics['mrr']:.4f}")
    print(f"Hits@1: {test_metrics['hits@1']:.4f}")
    print(f"Hits@3: {test_metrics['hits@3']:.4f}")
    print(f"Hits@10: {test_metrics['hits@10']:.4f}")

    wandb.finish()

if __name__ == "__main__":
    data_path = "/Users/andrasjoos/Documents/AI_masters/Period_9/ML4G/Project/LinkPred/data/"
    #drive_path = "/content/drive/MyDrive/trainandtest/"
    datasets = ['FB15k-237', 'WN18RR', 'YAGO3-10']
    for dataset in datasets:
        main(data_path,dataset)
