import torch
from pathlib import Path
import argparse
from models.conve import ConvE
from models.conve_deep import ConvEDeep
from models.conve_att import AttnConvE
from utils.preprocess import KGDataLoader, create_dataloader
from utils.train import train_conve, evaluate, evaluate_2pass
import wandb
import os
import datetime

def run_experiment(model_name, model_class, config, data_path, dataset, device):
    """Run a single experiment with given model and configuration"""
    
    # Initialize wandb
    run_name = f"{model_name}_{dataset}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    wandb.init(
        project="conve-kg",
        name=run_name,
        config=config,
        reinit=True
    )
    
    print(f"\nRunning {model_name} on {dataset}")
    print("Loading data...")
    data_loader = KGDataLoader(f"{data_path}{dataset}")
    datasets = data_loader.load_data()

    # Create dataloaders
    train_loader = create_dataloader(datasets['train'], batch_size=config['batch_size'], shuffle=True)
    valid_loader = create_dataloader(datasets['valid'], batch_size=config['batch_size'], shuffle=False)
    test_loader = create_dataloader(datasets['test'], batch_size=config['batch_size'], shuffle=False)

    # Initialize model
    print(f"Initializing {model_name} model...")
    model_params = {k: v for k, v in config.items() if k not in 
                   ['num_epochs', 'batch_size', 'learning_rate', 'label_smoothing', 'dataset']}
    model_params.update({
        'num_entities': len(data_loader.entity2id),
        'num_relations': len(data_loader.relation2id),
    })
    
    model = model_class(**model_params).to(device)
    print(f"Model is on device: {next(model.parameters()).device}")

    # Training parameters
    train_params = {
        'num_epochs': config['num_epochs'],
        'learning_rate': config['learning_rate'],
        'save_path': f"checkpoints/{model_name}_{dataset}",
        'eval_every': 1
    }

    # Train model
    print(f"Starting {model_name} training...")
    model = train_conve(
        model=model,
        train_dataloader=train_loader,
        valid_dataloader=valid_loader,
        device=device,
        **train_params
    )

    # Final evaluation
    print("\nEvaluating on test set...")
    test_metrics = evaluate_2pass(model, test_loader, device)

    # Log metrics
    wandb.log({
        'test_mr': test_metrics['mr'],
        'test_mrr': test_metrics['mrr'],
        'test_hits@1': test_metrics['hits@1'],
        'test_hits@3': test_metrics['hits@3'],
        'test_hits@10': test_metrics['hits@10']
    })

    print("\nFinal Test Metrics:")
    print(f"MR: {test_metrics['mr']:.1f}")
    print(f"MRR: {test_metrics['mrr']:.4f}")
    print(f"Hits@1: {test_metrics['hits@1']:.4f}")
    print(f"Hits@3: {test_metrics['hits@3']:.4f}")
    print(f"Hits@10: {test_metrics['hits@10']:.4f}")

    wandb.finish()

def main():
    wandb.finish()
    # Initialize wandb
    wandb_api_key = 'a15aa5a84ab821022d13f9aa3a59ec1770fe93a3'
    wandb.login(key=wandb_api_key)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available. Running on CPU!")
    else:
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")

    # Define configurations
    base_config = {
        "embedding_dim": 200,
        "embedding_shape1": 20,
        "num_epochs": 100,
        "batch_size": 128,
        "learning_rate": 0.003,
        "label_smoothing": 0.1,
    }

    model_configs = {
        "ConvE": {
            **base_config,
            "input_dropout": 0.2,
            "hidden_dropout": 0.3,
            "feature_map_dropout": 0.2,
        },
        "ConvEDeep": {
            **base_config,
            "input_dropout": 0.2,
            "hidden_dropout": 0.3,
            "feature_map_dropout": 0.2,
        },
        "AttnConvE": {
            **base_config,
            "num_attention_heads": 4,
            "ff_hidden_dim": 400,
            "conv_channels": 32,
            "dropout_attention": 0.1,
            "dropout_input": 0.2,
            "dropout_feature_map": 0.2,
        }
    }

    # Model classes
    models = {
        "ConvE": ConvE,
        "ConvEDeep": ConvEDeep,
        "AttnConvE": AttnConvE
    }

    # Datasets
    data_path = "/Users/andrasjoos/Documents/AI_masters/Period_9/ML4G/Project/LinkPred/data/"
    datasets = ['WN18RR', 'FB15k-237']

    # Run experiments
    embedding_settings = [True, False]  # True for stacked, False for alternating

    for use_stacked in embedding_settings:
        print(f"\nRunning experiments with {'stacked' if use_stacked else 'alternating'} embeddings")
        
        for dataset in datasets:
            for model_name, model_class in models.items():
                current_config = model_configs[model_name].copy()
                current_config['use_stacked_embeddings'] = use_stacked
                current_config['dataset'] = dataset
                
                try:
                    run_experiment(
                        model_name=f"{model_name}_{'stacked' if use_stacked else 'alt'}",
                        model_class=model_class,
                        config=current_config,
                        data_path=data_path,
                        dataset=dataset,
                        device=device
                    )
                except Exception as e:
                    print(f"Error running {model_name} on {dataset}: {str(e)}")
                    continue

if __name__ == "__main__":
    main()
