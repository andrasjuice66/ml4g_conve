import torch
from pathlib import Path
import argparse
from models.conve import ConvE
from utils.preprocess import KGDataLoader, create_dataloader
from LinkPred.utils.train import train_conve
import wandb

def main(args):
    # Initialize wandb
    wandb.init(
        project="conve-kg",
        config=args.__dict__
    )
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # Load datasets
    print("Loading data...")
    data_loader = KGDataLoader(args.data_dir)
    datasets = data_loader.load_data()
    
    # Create dataloaders
    train_loader = create_dataloader(
        datasets['train'], 
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    valid_loader = create_dataloader(
        datasets['valid'], 
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    test_loader = create_dataloader(
        datasets['test'], 
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # Initialize model
    print("Initializing model...")
    model = ConvE(
        num_entities=len(data_loader.entity2id),
        num_relations=len(data_loader.relation2id),
        embedding_dim=args.embedding_dim,
        embedding_shape1=args.embedding_shape1,
        input_dropout=args.input_dropout,
        hidden_dropout=args.hidden_dropout,
        feature_map_dropout=args.feature_map_dropout,
    )
    
    # Train model
    print("Starting training...")
    train_conve(
        model=model,
        train_dataloader=train_loader,
        valid_dataloader=valid_loader,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        device=device,
        label_smoothing=args.label_smoothing
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train ConvE model')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing train/valid/test files')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    
    # Model arguments
    parser.add_argument('--embedding_dim', type=int, default=200,
                        help='Dimension of embeddings')
    parser.add_argument('--embedding_shape1', type=int, default=20,
                        help='Reshape embedding to shape1 x shape2')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--label_smoothing', type=float, default=0.1,
                        help='Label smoothing factor')
    parser.add_argument('--input_dropout', type=float, default=0.2,
                        help='Input dropout rate')
    parser.add_argument('--hidden_dropout', type=float, default=0.3,
                        help='Hidden dropout rate')
    parser.add_argument('--feature_map_dropout', type=float, default=0.2,
                        help='Feature map dropout rate')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    main(args)
