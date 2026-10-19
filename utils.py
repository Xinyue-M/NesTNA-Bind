"""Residue pooling, metrics, dataset splits, and training bookkeeping."""

import random
import shutil
from pathlib import Path

import numpy as np
import torch
from sklearn import metrics
from torch_scatter import scatter_max

from dataloader import DATASETS, IterPatchDataset, read_pickle


# The existing surface-preparation scripts import this mapping from utils.
AA_CODES = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G',
    'HIS': 'H', 'LYS': 'K', 'ILE': 'I', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
    'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V',
    'TYR': 'Y', 'TRP': 'W', 'UNK': 'X', 'HID': 'H',
}


def log(directory, message):
    print(message, flush=True)
    with (Path(directory) / 'log.txt').open('a') as handle:
        handle.write(message + '\n')


def snapshot_scripts(directory):
    target = Path(directory) / 'scripts'
    target.mkdir()
    for source in Path(__file__).parent.glob('*.py'):
        shutil.copy2(str(source), str(target / source.name))


def pool_residues(probabilities, mapping, residue_count):
    """Return scores, winning patch indices, and coverage for every residue."""
    mapping = mapping.to(probabilities.device).reshape(-1)
    scores, winners = scatter_max(probabilities, mapping, dim=0, dim_size=residue_count)
    covered = torch.bincount(mapping, minlength=residue_count) > 0
    return scores, winners, covered


def compute_loss(criterion, output, sitelabel, pointlabel=None, patch_idx=None,
                 point2site=None, patch2site=None):
    """Keep the original loss signature and both residue-label metric definitions."""
    probabilities = output.softmax(dim=1)
    if point2site is None:
        return criterion(output, sitelabel), probabilities[:, 1]
    mapping = point2site.reshape(-1)
    count = len(sitelabel)
    legacy_scores, _, legacy_covered = pool_residues(probabilities[:, 1], patch2site, count)
    labels = torch.zeros_like(sitelabel)
    labels[mapping[pointlabel.reshape(-1) == 1].unique()] = 1
    scores, winners, covered = pool_residues(probabilities[:, 1], mapping[patch_idx[:, 0]], count)
    loss = criterion(probabilities[winners[covered]].log(), labels[covered])
    return loss, legacy_scores[legacy_covered], sitelabel[legacy_covered], scores[covered], labels[covered]


def binary_metrics(labels, probabilities, threshold=0.5):
    predictions = (probabilities > threshold).astype(np.int64)
    both_classes = np.unique(labels).size == 2
    result = {
        'precision': float(metrics.precision_score(labels, predictions, zero_division=0)),
        'recall': float(metrics.recall_score(labels, predictions, zero_division=0)),
        'accuracy': float(metrics.accuracy_score(labels, predictions)),
        'f1': float(metrics.f1_score(labels, predictions, zero_division=0)),
        'mcc': float(metrics.matthews_corrcoef(labels, predictions)),
        'auroc': float(metrics.roc_auc_score(labels, probabilities)) if both_classes else None,
        'auprc': None,
    }
    if both_classes:
        precision, recall, _ = metrics.precision_recall_curve(labels, probabilities)
        result['auprc'] = float(metrics.auc(recall, precision))
    return result


def select_threshold(labels, probabilities):
    # Same grid and first-best tie breaking as the original validation loop.
    thresholds = np.arange(1, 50) / 50
    scores = [metrics.f1_score(labels, probabilities > value, zero_division=0)
              for value in thresholds]
    return float(thresholds[int(np.argmax(scores))])


def split_files(files, ratio, fold):
    """Preserve the original per-call shuffle and fold slicing for comparability."""
    indices = list(range(len(files)))
    random.shuffle(indices)
    validation_size = int(len(files) * (1 - ratio)) + 1
    start, end = fold * validation_size, (fold + 1) * validation_size
    if end > len(files):
        raise ValueError('This split_ratio leaves too few proteins for the selected fold')
    validation = [files[index] for index in indices[start:end]]
    training = [files[index] for index in indices[:start] + indices[end:]]
    return training, validation


def load_data(args, kfold):
    train_name, test_name = DATASETS[args.taskname]
    base = Path(args.data_root) / 'pt_{}_{}_12_1'.format(args.num_sampling, args.num_neighbors)
    train_dir, test_dir = base / train_name, base / test_name
    train_esm = read_pickle(Path(args.data_root) / 'esm' / train_name)
    test_esm = read_pickle(Path(args.data_root) / 'esm' / test_name)
    files = [path.name for path in train_dir.iterdir() if path.is_file()]
    train_files, valid_files = split_files(files, args.split_ratio, kfold)
    datasets = [
        IterPatchDataset(train_dir, 'train', train_esm, train_files, args.num_neighbors),
        IterPatchDataset(train_dir, 'valid', train_esm, valid_files, args.num_neighbors),
        IterPatchDataset(test_dir, 'test', test_esm, num_neighbors=args.num_neighbors),
    ]
    log(args.checkpoints_dir, 'Fold {}: train={}, valid={}, test={}'.format(
        kfold, *(len(dataset) for dataset in datasets)))
    return datasets


class EarlyStopping:
    def __init__(self, directory, patience_stop=10, patience_lr=3, delta=0.0001):
        self.directory = Path(directory)
        self.patience_stop, self.patience_lr, self.delta = patience_stop, patience_lr, delta
        self.best_score, self.best_epoch, self.counter = None, None, 0

    def step(self, epoch, score, model, optimizer):
        if score is None or not np.isfinite(score):
            raise ValueError('Early stopping requires validation data with both label classes')
        if self.best_score is None or score > self.best_score + self.delta:
            self.best_score, self.best_epoch, self.counter = score, epoch, 0
            torch.save(model.state_dict(), str(self.directory / 'Bestmodel_Epoch{}.pth'.format(epoch)))
            log(self.directory, 'Saved best model at epoch {}'.format(epoch))
        else:
            self.counter += 1
            if self.counter == self.patience_lr:
                for group in optimizer.param_groups:
                    group['lr'] /= 10
                checkpoint = self.directory / 'Bestmodel_Epoch{}.pth'.format(self.best_epoch)
                model.load_state_dict(torch.load(str(checkpoint), map_location='cpu'))
                log(self.directory, 'Restored epoch {}; learning rate={}'.format(
                    self.best_epoch, optimizer.param_groups[0]['lr']))
        return self.counter >= self.patience_stop
