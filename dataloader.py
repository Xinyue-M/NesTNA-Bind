"""Shared protein preparation for training and label-free inference."""

import pickle
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset


DATA_ROOT = Path(__file__).resolve().parent.parent
DATASETS = {'RNA': ('RNA-663_Train', 'RNA-157_Test'),
            'DNA': ('DNA-719_Train', 'DNA-179_Test'),
            'RNA_our': ('RNA_our_Train', 'RNA_our_Test'),
            'PPI': ('training_ppi', 'testing_ppi')}


@lru_cache(maxsize=2)
def read_pickle(path):
    """Cache the two large ESM dictionaries across folds."""
    with Path(path).open('rb') as handle:
        return pickle.load(handle)


def farthest_point_sample(xyz, count):
    batch_size, points, _ = xyz.shape
    selected = torch.zeros(batch_size, count, dtype=torch.long, device=xyz.device)
    distance = torch.full((batch_size, points), 1e10, device=xyz.device)
    farthest = torch.randint(points, (batch_size,)).to(xyz.device)
    batch = torch.arange(batch_size, device=xyz.device)
    for index in range(count):
        selected[:, index] = farthest
        centers = xyz[batch, farthest][:, None]
        distance = torch.minimum(distance, ((xyz - centers) ** 2).sum(-1))
        farthest = distance.max(-1).indices
    return selected


@dataclass
class Protein:
    name: str
    features: torch.Tensor
    xyz: torch.Tensor
    esm: torch.Tensor
    patch_idx: torch.Tensor
    point2site: torch.Tensor
    residue_ids: list
    site_labels: torch.Tensor = None
    original_labels: torch.Tensor = None
    majority2site: torch.Tensor = None
    point_labels: torch.Tensor = None

    @property
    def centroid2site(self):
        return self.point2site[self.patch_idx[:, 0]]

    def inputs(self, device):
        return self.features.to(device), self.xyz.to(device), None, self.esm.to(device)


def prepare_protein(name, data, esm_entry, max_patches=8000, num_neighbors=128,
                    with_labels=False):
    """Construct [patch,neighbor,feature] tensors; labels never enter model inputs."""
    xyz = torch.as_tensor(data['xyz'], dtype=torch.float32)
    mapping = torch.as_tensor(data['point2site'], dtype=torch.long).reshape(-1)
    patches = torch.as_tensor(data['patch_idx'], dtype=torch.long)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) != len(mapping):
        raise ValueError('Expected xyz [V,3] and point2site [V] or [V,1]')
    if not len(mapping) or mapping.min() < 0:
        raise ValueError('point2site must contain nonnegative indices')
    if patches.ndim != 2 or patches.shape[1] != num_neighbors or not len(patches):
        raise ValueError('Expected nonempty patch_idx [P,{}]'.format(num_neighbors))
    if patches.min() < 0 or patches.max() >= len(xyz):
        raise ValueError('patch_idx is out of range')
    if len(patches) > max_patches:
        # All supplied preprocessed surfaces have one patch per point.
        if len(patches) != len(xyz):
            raise ValueError('FPS requires one patch per surface point')
        patches = patches[farthest_point_sample(xyz.unsqueeze(0), max_patches).reshape(-1)]
    residue_ids = list(data.get('res_idx2id', range(int(mapping.max()) + 1)))
    if mapping.max() >= len(residue_ids):
        raise ValueError('point2site exceeds the residue count')
    centroid2site = mapping[patches[:, 0]]
    embeddings = torch.as_tensor(esm_entry[-1], dtype=torch.float32)
    if embeddings.ndim == 3 and embeddings.shape[0] == 1:
        embeddings = embeddings[0]
    if embeddings.ndim != 2 or embeddings.shape[1] != 1280 or centroid2site.max() >= len(embeddings):
        raise ValueError('Expected ESM [L,1280] covering all patch-center residues')

    def feature(key):
        values = torch.as_tensor(data[key], dtype=torch.float32)
        if values.shape != (len(xyz), 3):
            raise ValueError('{} must have shape [V,3]'.format(key))
        return values[patches]

    atom_types = torch.as_tensor(data['atom_type'], dtype=torch.long).reshape(-1)
    if len(atom_types) != len(xyz):
        raise ValueError('Expected one atom type per surface point')
    features = torch.cat((feature('chem'), feature('geo'),
                          F.one_hot(atom_types[patches], 6).float()), dim=-1)
    protein = Protein(name, features, xyz[patches], embeddings[centroid2site],
                      patches, mapping, residue_ids)
    if with_labels:
        labels = torch.as_tensor(data['interface_label'], dtype=torch.long).reshape(-1)
        if len(labels) != len(mapping) or not ((labels == 0) | (labels == 1)).all():
            raise ValueError('interface_label must contain one binary value per point')
        protein.site_labels = torch.zeros(len(residue_ids), dtype=torch.long)
        protein.point_labels = labels
        protein.site_labels[mapping[labels == 1].unique()] = 1
        if 'site_label' in data:
            protein.original_labels = torch.as_tensor(data['site_label'], dtype=torch.long)
            if protein.original_labels.shape != (len(residue_ids),):
                raise ValueError('site_label and res_idx2id lengths differ')
            protein.majority2site = torch.stack([
                torch.bincount(indices).argmax() for indices in mapping[patches]
            ])
    return protein


class IterPatchDataset(IterableDataset):
    def __init__(self, file_dir, set_type, esm_info, file_list=None, num_neighbors=128):
        super().__init__()
        self.file_dir = Path(file_dir)
        self.file_list = list(file_list) if file_list is not None else [
            path.name for path in self.file_dir.iterdir() if path.is_file()]
        self.esm_info = esm_info
        self.max_patches = 2048 if set_type == 'train' else 8000
        self.num_neighbors = num_neighbors

    def __len__(self):
        return len(self.file_list)

    def __iter__(self):
        for name in self.file_list:
            if name not in self.esm_info:
                print('Skipping {}: no ESM entry'.format(name), flush=True)
                continue
            with (self.file_dir / name).open('rb') as handle:
                data = pickle.load(handle)
            try:
                yield prepare_protein(name, data, self.esm_info[name], self.max_patches,
                                      self.num_neighbors, with_labels=True)
            except (KeyError, ValueError, IndexError) as error:
                raise ValueError('{}: {}'.format(name, error)) from error
