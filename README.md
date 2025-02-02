# ConvE reproduction for Link Prediction in Knowledge Graphs

This repository implements ConvE and its variants (DeepConvE, AttnConvE) for knowledge graph link prediction in the context of the ML4G course.

## Setup

1. Clone the repository:
```bash
git clone https://github.com/andrasjuice66/ml4g_conve.git
cd ml4g_conve
```

2. Install requirements:
```bash
pip install torch numpy wandb
```

3. Set up Weights & Biases:
```bash
export WANDB_API_KEY=your_api_key_here
```
Or add to your shell profile (~/.bashrc, ~/.zshrc, etc.)

4. Preprocess the datasets:
```bash
python utils/create_dataset.py WN18RR
python utils/create_dataset.py FB15K-237
```
This creates JSON files with integer IDs for entities and relations.

5. Configure your run:
- Edit `config` dictionary in `main_all.py`
- Adjust hyperparameters as needed

6. Run training:
```bash
python main_all.py
```

The code will execute a grid run on all the combinations of datasets, models, embedding styles.
