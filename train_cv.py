"""Train with one shared epoch loop for training, validation, and testing."""

import csv
import datetime
import json
import random
import numpy as np
import torch
from tqdm import tqdm

from Arguments import parser
from utils import (EarlyStopping, binary_metrics, compute_loss, load_data,
                   log, select_threshold, snapshot_scripts)
from vit_esm import build_model


def run_epoch(model, dataset, device, taskbase='site', optimizer=None, description='Evaluate'):
    training = optimizer is not None
    model.train(training)
    losses, probabilities, labels, legacy_scores, legacy_labels = [], [], [], [], []
    criterion = torch.nn.NLLLoss() if taskbase == 'site' else torch.nn.CrossEntropyLoss()
    progress = tqdm(dataset, desc=description, leave=False)
    with torch.set_grad_enabled(training):
        for protein in progress:
            logits = model(*protein.inputs(device)).reshape(-1, 2)
            if protein.original_labels is None:
                raise ValueError('{}: training/evaluation requires site_label'.format(protein.name))
            targets = protein.original_labels.to(device)
            if taskbase == 'site':
                point_labels = protein.point_labels.to(device)
                loss, old_scores, old_labels, scores, truth = compute_loss(
                    criterion, logits, targets, point_labels, protein.patch_idx.to(device),
                    protein.point2site.to(device), protein.majority2site.to(device))
                legacy_scores.append(old_scores.detach().cpu().numpy())
                legacy_labels.append(old_labels.cpu().numpy())
            else:
                truth = targets[protein.majority2site.to(device)]
                loss, scores = compute_loss(criterion, logits, truth)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            losses.append(loss.item())
            probabilities.append(scores.detach().cpu().numpy())
            labels.append(truth.cpu().numpy())
            progress.set_postfix(loss=loss.item())
    if not losses:
        raise ValueError('No usable proteins in {}'.format(description))
    result = dict(loss=float(np.mean(losses)), probabilities=np.concatenate(probabilities),
                  labels=np.concatenate(labels))
    if legacy_scores:
        result.update(legacy_probabilities=np.concatenate(legacy_scores),
                      legacy_labels=np.concatenate(legacy_labels))
    return result


def evaluate(result, threshold):
    return binary_metrics(result['labels'], result['probabilities'], threshold)


def onefold(args, training, validation, testing):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(args)
    if args.load_model_path:
        model.load_state_dict(torch.load(str(args.load_model_path), map_location='cpu'))
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, amsgrad=True)
    early_stop = EarlyStopping(args.checkpoints_dir)
    with (args.checkpoints_dir / 'out.csv').open('w', newline='') as handle:
        writer = None
        for epoch in range(args.start_epoch, args.epochs):
            train = run_epoch(model, training, device, args.taskbase, optimizer, 'Train {}'.format(epoch))
            valid = run_epoch(model, validation, device, args.taskbase, description='Valid {}'.format(epoch))
            test = run_epoch(model, testing, device, args.taskbase,
                             description='Test {}'.format(epoch))
            threshold = select_threshold(valid['labels'], valid['probabilities'])
            row = {'Epoch': epoch, 'threshold': threshold}
            for name, result in [('train', train), ('val', valid), ('test', test)]:
                row[name + '_loss'] = result['loss']
                row.update({name + '_' + key: value for key, value in evaluate(result, threshold).items()})
            if 'legacy_labels' in valid:
                legacy_threshold = select_threshold(valid['legacy_labels'], valid['legacy_probabilities'])
                row['legacy_threshold'] = legacy_threshold
                for name, result in [('val', valid), ('test', test)]:
                    values = binary_metrics(result['legacy_labels'], result['legacy_probabilities'], legacy_threshold)
                    row.update({'legacy_' + name + '_' + key: value for key, value in values.items()})
            if writer is None:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            handle.flush()
            log(args.checkpoints_dir, json.dumps(row))
            torch.save(model.state_dict(), str(args.checkpoints_dir / 'Model_Epoch{}.pth'.format(epoch)))
            torch.save([test['probabilities'], test['labels']],
                       str(args.checkpoints_dir / 'Predict_Epoch{}.pth'.format(epoch)))
            if early_stop.step(epoch, row['val_auroc'], model, optimizer):
                break


def main():
    args = parser.parse_args()
    if not 0 < args.split_ratio < 1 or not 0 <= args.start_epoch < args.epochs:
        parser.error('Require 0 < split_ratio < 1 and 0 <= start_epoch < epochs')
    if args.num_neighbors < 16 or args.n_layers < 1 or args.lr <= 0 or not 0 <= args.dropout < 1:
        parser.error('Invalid neighbor count, model depth, learning rate, or dropout')
    # Preserve the original split/RNG protocol during this refactor.
    random.seed(123)
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')
    run = args.checkpoints_dir / '{}_{}_{}'.format(timestamp, args.taskbase, args.taskname)
    for fold in range(5) if args.fold is None else [args.fold]:
        args.checkpoints_dir = run / 'fold_{}'.format(fold)
        args.checkpoints_dir.mkdir(parents=True)
        snapshot_scripts(args.checkpoints_dir)
        log(args.checkpoints_dir, json.dumps(vars(args), default=str, sort_keys=True))
        datasets = load_data(args, fold)
        with (args.checkpoints_dir / 'split.json').open('w') as handle:
            json.dump(dict(zip(['train', 'validation', 'test'],
                               [dataset.file_list for dataset in datasets])), handle, indent=2)
        onefold(args, *datasets)


if __name__ == '__main__':
    main()
