import torch
import numpy as np
from pathlib import Path
import wandb
from models.conve import ConvE
from models.conve_deep import DeepConvE
from models.conve_att import AttnConvE
from utils.train import train, evaluation
from torch.utils.data import DataLoader
from utils.preprocess import LinkPredictionDataset, collate_fn, build_vocab
from datetime import datetime


config = {
    'batch_size': 128,
    'test_batch_size': 128,
    'epochs': 100,
    'lr': 0.003,
    'seed': 17,
    'log_interval': 100,
    'l2': 0.0,
    'embedding_dim': 200,
    'embedding_shape1': 20,
    'hidden_drop': 0.3,
    'input_drop': 0.2,
    'feat_drop': 0.2,
    'lr_decay': 0.995,
    'loader_threads': 4,
    'use_bias': False,
    'label_smoothing': 0.1,
    'hidden_size': 9728,
    'num_attention_heads': 4,
    'ff_hidden_dim': 200,
    'dropout_attention': 0.1,
    'dropout_input': 0.2,
    'dropout_feature': 0.2,


    #RUN CONFIGS
    'wandb_project': 'final-kg-runs',
    'wandb_api_key': 'a15aa5a84ab821022d13f9aa3a59ec1770fe93a3',
    'datasets': ['WN18RR', 'FB15K-237'],# 'YAGO3-10'],  
    'models': ['deepconve', 'attnconve', 'conve'],
    'embedding_style': ['stacked', 'alternating'],
    'path': '/Users/andrasjoos/Documents/AI_masters/Period_9/ML4G/Project/LinkPred/data'
}

def main():
    wandb.finish()
    wandb.login(key=config['wandb_api_key']) 
    
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    for dataset in config['datasets']:
        for model_name in config['models']:
            for embedding_style in config['embedding_style']:
                if (model_name == 'attnconve' or model_name == 'deepconve') and embedding_style == 'alternating': 
                    continue
                current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                run = wandb.init(
                    project=config['wandb_project'],
                    config=config,
                    name=f"{model_name}_{dataset}_{embedding_style}_{current_time}",
                    reinit=True  
                )

                print(f"[INFO] Training {model_name} on {dataset} dataset")
                
                base_path = f"{config['path']}/{dataset}"
                entity2id_path = str(Path(base_path) / "entity.json")
                relation2id_path = str(Path(base_path) / "relation.json")
                train_json = str(Path(base_path) / "train.json")
                valid_json = str(Path(base_path) / "valid.json")
                test_json = str(Path(base_path) / "test.json")

                vocab = build_vocab(entity2id_path, relation2id_path)
                num_entities = vocab['e1'].num_token
                num_relations = vocab['rel'].num_token

                train_dataset = LinkPredictionDataset(train_json, mode='train')
                valid_dataset = LinkPredictionDataset(valid_json, mode='eval')
                test_dataset = LinkPredictionDataset(test_json, mode='eval')

                train_loader = DataLoader(
                    dataset=train_dataset,
                    batch_size=config['batch_size'],
                    shuffle=True,
                    num_workers=0,
                    collate_fn=collate_fn
                )
                val_loader = DataLoader(
                    dataset=valid_dataset,
                    batch_size=config['test_batch_size'],
                    shuffle=False,
                    num_workers=0,
                    collate_fn=collate_fn
                )
                test_loader = DataLoader(
                    dataset=test_dataset,
                    batch_size=config['test_batch_size'],
                    shuffle=False,
                    num_workers=0,
                    collate_fn=collate_fn
                )

                if model_name == 'conve':
                    model = ConvE(config, num_entities, num_relations, embedding_style)
                elif model_name == 'deepconve':
                    model = DeepConvE(config, num_entities, num_relations)
                elif model_name == 'attnconve':
                    model = AttnConvE(config, num_entities, num_relations)

                model.to(device)
                model.init()

                optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['l2'])

                print("Training...")
                train(model, config, train_loader, num_entities, val_loader, optimizer, device)
                print("Evaluating on the test set...")
                _, test_wandb_metrics = evaluation(model, test_loader)
                
                wandb.log({"test_" + k: v for k, v in test_wandb_metrics.items()})
                
                run.finish()

if __name__ == "__main__":
    main()