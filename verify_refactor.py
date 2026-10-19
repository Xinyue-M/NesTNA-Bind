"""CPU regression checks against the original code and real RNA protein files."""

import ast
import csv
import importlib.util
import json
import os
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch_scatter import scatter_max

from dataloader import DATA_ROOT, IterPatchDataset, prepare_protein, read_pickle
from inference import seed_protein
from train_cv import onefold
from utils import EarlyStopping, compute_loss, pool_residues
from vit_esm import ViT_ESM


def reference_module(name):
    spec = importlib.util.spec_from_file_location('reference_' + name, str(DATA_ROOT / (name + '.py')))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reference_loss():
    tree = ast.parse((DATA_ROOT / 'utils.py').read_text())
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'compute_loss']
    scope = dict(torch=torch, scatter_max=scatter_max)
    exec(compile(tree, 'original_compute_loss', 'exec'), scope)
    return scope['compute_loss']


def compare(left, right):
    torch.testing.assert_close(left, right, rtol=0, atol=0)


def main():
    torch.set_num_threads(4)
    checkpoint = DATA_ROOT / 'logs/pbs_11176288.node1/2026-10-19_22-06_site_RNA_vertices_all_128/Bestmodel_Epoch0.pth'
    old_model = reference_module('vit_esm').ViT_ESM(
        num_classes=2, patch_size=128, dim=32, depth=2, heads=8, mlp_dim=64,
        channels=12, hidden_dim=256, dropout=0.1)
    new_model = ViT_ESM()
    weights = torch.load(str(checkpoint), map_location='cpu')
    old_model.load_state_dict(weights)
    new_model.load_state_dict(weights)
    assert list(old_model.state_dict()) == list(new_model.state_dict())
    torch.manual_seed(12)
    features, xyz, esm = torch.randn(6, 128, 12), torch.randn(6, 128, 3), torch.randn(6, 1280)
    largest_gradient_difference = 0.
    for training in (False, True):
        old_model.train(training)
        new_model.train(training)
        torch.manual_seed(21)
        old_output = old_model(features, xyz, None, esm)
        torch.manual_seed(21)
        new_output = new_model(features, xyz, None, esm)
        compare(old_output, new_output)
        old_output.square().mean().backward()
        new_output.square().mean().backward()
        for (name, old), (_, new) in zip(old_model.named_parameters(), new_model.named_parameters()):
            assert (old.grad is None) == (new.grad is None), name
            if old.grad is not None:
                largest_gradient_difference = max(
                    largest_gradient_difference, float((old.grad - new.grad).abs().max()))
                torch.testing.assert_close(old.grad, new.grad, rtol=1e-5, atol=1e-6)
        old_model.zero_grad()
        new_model.zero_grad()
    print('PASS: strict checkpoint load, exact eval/train outputs; max gradient difference={}'.format(
        largest_gradient_difference), flush=True)

    cache = read_pickle(DATA_ROOT / 'esm/RNA-157_Test')
    directory = DATA_ROOT / 'pt_all_128_12_1/RNA-157_Test'
    old_loader = reference_module('dataloader')
    old_compute_loss = reference_loss()
    data = None
    for name, mode in [('1b2m_A_C', 'test'), ('1fje_B_A', 'train')]:
        with (directory / name).open('rb') as handle:
            data = pickle.load(handle)
        torch.manual_seed(31)
        original = next(iter(old_loader.IterPatchDataset(str(directory), mode, cache, [name])))
        torch.manual_seed(31)
        updated = next(iter(IterPatchDataset(directory, mode, cache, [name])))
        for old, new in [(original[1].float(), updated.features),
                         (original[2].float(), updated.xyz), (original[4], updated.esm),
                         (original[9], updated.patch_idx), (original[10], updated.majority2site)]:
            compare(old, new)
        logits = torch.randn(len(updated.patch_idx), 2, requires_grad=True)
        cloned_logits = logits.detach().clone().requires_grad_()
        criterion = torch.nn.NLLLoss()
        original_loss = old_compute_loss(criterion, logits, original[7], original[6],
                                         original[9], original[8], original[10])
        updated_loss = compute_loss(criterion, cloned_logits, updated.original_labels,
                                    updated.point_labels, updated.patch_idx,
                                    updated.point2site, updated.majority2site)
        for left, right in zip(original_loss, updated_loss):
            compare(left, right)
        original_loss[0].backward()
        updated_loss[0].backward()
        compare(logits.grad, cloned_logits.grad)
    print('PASS: real protein loading, FPS, labels, both metric mappings, loss and loss gradients', flush=True)

    name = '1b2m_A_C'
    with (directory / name).open('rb') as handle:
        data = pickle.load(handle)
    seed_protein(0, name)
    protein = prepare_protein(name, data, cache[name], with_labels=False)
    new_model.eval()
    with torch.no_grad():
        scores, _, covered = pool_residues(new_model(*protein.inputs('cpu')).reshape(-1, 2).softmax(1)[:, 1],
                                          protein.centroid2site, len(protein.residue_ids))
    baseline = DATA_ROOT / 'predictions/smoke_labeled/residue_predictions.csv'
    with baseline.open() as handle:
        rows = [row for row in csv.DictReader(handle) if row['protein_id'] == name]
    np.testing.assert_array_equal(scores[covered].numpy(), np.array([float(row['probability']) for row in rows], dtype=np.float32))
    for field in ['site_label', 'interface_label', 'nuv', 'seq']:
        data.pop(field, None)
    unlabeled = prepare_protein(name, data, cache[name])
    compare(protein.features, unlabeled.features)
    compare(protein.esm, unlabeled.esm)
    assert unlabeled.site_labels is None
    print('PASS: previous inference CSV probabilities and unlabeled inputs', flush=True)

    with (directory / name).open('rb') as handle:
        data = pickle.load(handle)
    sample = prepare_protein(name, data, cache[name], max_patches=96, with_labels=True)
    assert len(sample.site_labels[sample.centroid2site].unique()) == 2
    with tempfile.TemporaryDirectory(prefix='nestnabind_refactor_') as temporary:
        args = SimpleNamespace(n_layers=2, dropout=0.1, num_neighbors=128, load_model_path=None,
                               lr=0.001, checkpoints_dir=Path(temporary), taskbase='site', start_epoch=0, epochs=1)
        onefold(args, [sample], [sample], [sample])
        with (args.checkpoints_dir / 'out.csv').open() as handle:
            assert len(list(csv.DictReader(handle))) == 1
        new_model.load_state_dict(torch.load(str(args.checkpoints_dir / 'Bestmodel_Epoch0.pth')))
        probabilities, labels = torch.load(str(args.checkpoints_dir / 'Predict_Epoch0.pth'))
        assert probabilities.shape == labels.shape and np.isfinite(probabilities).all()
        # Exercise the formerly broken first-best-epoch LR reduction branch.
        stopper = EarlyStopping(args.checkpoints_dir)
        optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001)
        for epoch, score in enumerate([0.8, 0.7, 0.6, 0.5]):
            stopper.step(epoch, score, new_model, optimizer)
        assert stopper.best_epoch == 0 and optimizer.param_groups[0]['lr'] == 0.0001
    print('PASS: one-epoch train/validation/test, saved artifacts, early stopping', flush=True)
    # Also run the real CLI and default dataset layout on a small fixture.
    with tempfile.TemporaryDirectory(prefix='nestnabind_cli_') as temporary:
        fixture = Path(temporary)
        raw = dict(data)
        indices = torch.linspace(0, len(raw['patch_idx']) - 1, 96).long()
        raw['patch_idx'] = raw['patch_idx'][indices]
        for dataset_name, count in [('RNA-663_Train', 4), ('RNA-157_Test', 2)]:
            folder = fixture / 'pt_all_128_12_1' / dataset_name
            folder.mkdir(parents=True)
            entries = {}
            for index in range(count):
                filename = '{}_{}'.format(name, index)
                with (folder / filename).open('wb') as handle:
                    pickle.dump(raw, handle, pickle.HIGHEST_PROTOCOL)
                entries[filename] = cache[name]
            (fixture / 'esm').mkdir(exist_ok=True)
            with (fixture / 'esm' / dataset_name).open('wb') as handle:
                pickle.dump(entries, handle, pickle.HIGHEST_PROTOCOL)
        environment = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
        subprocess.run([
            sys.executable, str(Path(__file__).resolve().parent / 'train_cv.py'),
            '--taskname', 'RNA', '--taskbase', 'site', '--num_sampling', 'all',
            '--data-root', str(fixture), '--checkpoints_dir', str(fixture / 'runs'),
            '--fold', '0', '--epochs', '1',
        ], cwd=temporary, env=environment, check=True)
        outputs = list((fixture / 'runs').glob('*/fold_0'))
        assert len(outputs) == 1
        with (outputs[0] / 'split.json').open() as handle:
            split = json.load(handle)
        assert [len(split[key]) for key in ['train', 'validation', 'test']] == [3, 1, 2]
        assert not set(split['train']) & set(split['validation'])
        assert (outputs[0] / 'Bestmodel_Epoch0.pth').is_file()
    print('PASS: training CLI, custom data root, split manifest, isolated fold outputs', flush=True)
    print('ALL_REFACTOR_CHECKS_PASSED', flush=True)


if __name__ == '__main__':
    main()
