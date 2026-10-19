"""Training options and explicitly labeled legacy metadata."""

import argparse
from pathlib import Path
from dataloader import DATA_ROOT, DATASETS


def make_parser():
    parser = argparse.ArgumentParser(description='Train the surface/ESM binding-site model')
    parser.add_argument('--taskname', choices=sorted(DATASETS), default='RNA')
    parser.add_argument('--taskbase', choices=['site', 'patch'], default='patch')
    parser.add_argument('--num_sampling', default=1024)
    parser.add_argument('--num_neighbors', type=int, default=128)
    parser.add_argument('--split_ratio', type=float, default=0.8)
    parser.add_argument('--fold', type=int, choices=range(5), default=None)
    parser.add_argument('--data-root', type=Path, default=DATA_ROOT)
    parser.add_argument('--checkpoints_dir', type=Path, default=Path(__file__).parent / 'logs')
    parser.add_argument('--load_model_path', type=Path)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--nclasses', type=int, choices=[2], default=2)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--loss_type', default='vertices', help='Legacy metadata; taskbase selects the loss')
    parser.add_argument('--seed', type=int, default=42,
                        help='Legacy metadata; Python shuffle seed stays 123 for compatibility')
    return parser


parser = make_parser()
