"""Predict binding probabilities for preprocessed proteins and cached ESM features."""

import argparse
import csv
import hashlib
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch

from dataloader import DATA_ROOT, prepare_protein, read_pickle
from utils import binary_metrics, pool_residues
from vit_esm import build_model


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', '--load_model_path', required=True, type=Path)
    parser.add_argument('--data-dir', type=Path, default=DATA_ROOT / 'pt_all_128_12_1/RNA-157_Test')
    parser.add_argument('--esm-file', type=Path, default=DATA_ROOT / 'esm/RNA-157_Test')
    parser.add_argument('--protein', nargs='+', help='Exact filenames; omit for the whole directory')
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).parent / 'predictions')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--max-patches', type=int, default=8000)
    parser.add_argument('--num-neighbors', type=int, default=128)
    parser.add_argument('--n-layers', type=int, default=2)
    parser.add_argument('--device', default='auto', help='auto, cpu, cuda, or cuda:0, etc.')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--evaluate', action='store_true', help='Evaluate using interface-derived site labels')
    parser.set_defaults(dropout=0.1)
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1 or args.max_patches < 2 or args.num_neighbors < 16:
        parser.error('Require threshold in [0,1], max-patches >= 2, num-neighbors >= 16')
    if args.n_layers < 1 or args.threads < 1:
        parser.error('Require positive n-layers and threads')
    for path in (args.checkpoint, args.esm_file):
        if not path.is_file():
            parser.error('File not found: {}'.format(path))
    if not args.data_dir.is_dir():
        parser.error('Data directory not found: {}'.format(args.data_dir))
    return args


def seed_protein(seed, name):
    offset = int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], 'little')
    value = (seed + offset) % (2 ** 32)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def predict(model, protein, device):
    logits = model(*protein.inputs(device)).reshape(-1, 2)
    scores, _, covered = pool_residues(logits.softmax(1)[:, 1],
                                      protein.centroid2site, len(protein.residue_ids))
    if not torch.isfinite(scores[covered]).all():
        raise ValueError('{}: model returned nonfinite probabilities'.format(protein.name))
    return scores.cpu().numpy(), covered.cpu().numpy()


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    requested_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(requested_device if args.device == 'auto' else args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable; use --device cpu')
    names = args.protein or sorted(path.name for path in args.data_dir.iterdir() if path.is_file())
    if not names or len(names) != len(set(names)):
        raise ValueError('Select at least one protein, without duplicates')
    for name in names:
        if Path(name).name != name or not (args.data_dir / name).is_file():
            raise ValueError('Protein file not found: {}'.format(name))
    started = time.time()
    print('Loading checkpoint {} on {}'.format(args.checkpoint, device), flush=True)
    model = build_model(args)
    model.load_state_dict(torch.load(str(args.checkpoint), map_location='cpu'))
    model.to(device).eval()
    print('Loading ESM cache: {}'.format(args.esm_file), flush=True)
    esm_info = read_pickle(args.esm_file)
    missing = [name for name in names if name not in esm_info]
    if missing:
        raise ValueError('Missing ESM entries: {}'.format(', '.join(missing)))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    temporary = args.output_dir / 'residue_predictions.csv.tmp'
    summaries, all_labels, all_probabilities = [], [], []
    with temporary.open('w', newline='') as handle, torch.no_grad():
        writer = csv.writer(handle)
        writer.writerow(['protein_id', 'site_index', 'residue_id', 'probability',
                         'prediction', 'covered', 'patch_count', 'interface_site_label'])
        for index, name in enumerate(names):
            seed_protein(args.seed, name)
            with (args.data_dir / name).open('rb') as source:
                protein = prepare_protein(name, pickle.load(source), esm_info[name],
                                          args.max_patches, args.num_neighbors, args.evaluate)
            probabilities, covered = predict(model, protein, device)
            counts = np.bincount(protein.centroid2site.numpy(), minlength=len(covered))
            for site, residue_id in enumerate(protein.residue_ids):
                writer.writerow([
                    name, site, residue_id, float(probabilities[site]) if covered[site] else '',
                    int(probabilities[site] > args.threshold) if covered[site] else '',
                    int(covered[site]), int(counts[site]),
                    int(protein.site_labels[site]) if args.evaluate else '',
                ])
            summary = dict(protein_id=name, patches=int(counts.sum()),
                           residues=len(covered), covered_residues=int(covered.sum()))
            if args.evaluate:
                labels = protein.site_labels.numpy()[covered]
                all_labels.append(labels)
                all_probabilities.append(probabilities[covered])
                summary['metrics'] = binary_metrics(labels, probabilities[covered], args.threshold)
            summaries.append(summary)
            print('[{}/{}] {}: {}/{} residues covered'.format(
                index + 1, len(names), name, int(covered.sum()), len(covered)), flush=True)
    output = args.output_dir / 'residue_predictions.csv'
    temporary.replace(output)
    result = dict(checkpoint=str(args.checkpoint.resolve()), data_dir=str(args.data_dir.resolve()),
                  esm_file=str(args.esm_file.resolve()), device=str(device), seed=args.seed,
                  threshold=args.threshold, max_patches=args.max_patches, num_neighbors=args.num_neighbors,
                  n_layers=args.n_layers, aggregation='max positive patch probability per centroid residue',
                  protein_count=len(names), proteins=summaries, elapsed_seconds=time.time() - started)
    if args.evaluate:
        result['evaluation_labels'] = 'any positive interface point in residue'
        result['metrics'] = binary_metrics(np.concatenate(all_labels),
                                           np.concatenate(all_probabilities), args.threshold)
    with (args.output_dir / 'summary.json').open('w') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print('Saved predictions and summary to {}'.format(args.output_dir.resolve()), flush=True)


if __name__ == '__main__':
    main()
