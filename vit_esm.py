"""Local/global surface transformer, compatible with existing ViT_ESM weights."""

import torch
from torch import nn
from einops import rearrange


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, hidden_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, dim), nn.Dropout(dropout),
        )

    def forward(self, features):
        return self.net(features)


class Attention(nn.Module):
    def __init__(self, input_dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(input_dim, heads * dim_head * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(heads * dim_head, input_dim), nn.Dropout(dropout))
        if heads == 1 and dim_head == input_dim:
            self.to_out = nn.Identity()

    def forward(self, features):
        query, key, value = (
            rearrange(part, 'b n (h d) -> b h n d', h=self.heads)
            for part in self.to_qkv(self.norm(features)).chunk(3, dim=-1)
        )
        weights = self.dropout(self.attend((query @ key.transpose(-1, -2)) * self.scale))
        return self.to_out(rearrange(weights @ value, 'b h n d -> b n (h d)'))


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([
            nn.ModuleList([Attention(dim, heads, dim_head, dropout),
                           FeedForward(dim, mlp_dim, dropout)])
            for _ in range(depth)
        ])

    def forward(self, features):
        for attention, feed_forward in self.layers:
            features = attention(features) + features
            features = feed_forward(features) + features
        return self.norm(features)


class ViT_ESM(nn.Module):
    def __init__(self, num_classes=2, dim=32, depth=2, heads=8, mlp_dim=64,
                 hidden_dim=256, patch_size=128, channels=12, dropout=0.1):
        super().__init__()
        if num_classes != 2:
            raise ValueError('Existing checkpoints use two output classes')
        self.channel_embedding = nn.Sequential(
            nn.Linear(channels, 32), nn.LeakyReLU(0.2),
            nn.Linear(32, 32), nn.LeakyReLU(0.2),
        )
        # Keep this inactive layer so old state dictionaries load strictly.
        self.norm_in = nn.LayerNorm(128)
        self.sub_neighbor = self.sub_patch = 16
        self.to_subpatch_embedding = nn.Sequential(
            nn.LayerNorm(self.sub_neighbor * 32),
            nn.Linear(self.sub_neighbor * 32, dim), nn.LayerNorm(dim),
        )
        self.sub_pool = nn.MaxPool1d(self.sub_patch, self.sub_patch)
        self.sub_transformer = Transformer(dim, depth, heads, 8, mlp_dim, dropout)
        self.esm_linear = nn.Linear(1280, dim)
        self.pos_embedding = nn.Sequential(nn.Linear(3, dim), nn.GELU(), nn.Linear(dim, dim))
        self.dropout = nn.Dropout(dropout)
        self.transformer = Transformer(dim, depth, heads, 8, mlp_dim, dropout)
        self.ln = nn.LayerNorm(dim * 2)
        self.predict_head = nn.Sequential(
            nn.Linear(dim * 2, 16), nn.LeakyReLU(0.2),
            nn.Linear(16, 16), nn.LeakyReLU(0.2), nn.Linear(16, 2),
        )

    def _subpatch_indices(self, xyz):
        """FPS centers and their nearest neighbors inside each surface patch."""
        patches, points, _ = xyz.shape
        batch = torch.arange(patches, device=xyz.device)
        distance = torch.full((patches, points), 1e5, device=xyz.device)
        # Preserve the original sampling range for numerical comparisons.
        farthest = torch.randint(0, points - 1, (patches,)).to(xyz.device)
        neighbors = []
        for _ in range(self.sub_patch):
            centers = xyz[batch, farthest]
            squared_distance = ((xyz - centers[:, None]) ** 2).sum(-1)
            distance = torch.minimum(distance, squared_distance)
            farthest = distance.max(-1).indices
            neighbors.append(squared_distance.topk(self.sub_neighbor, largest=False).indices)
        return torch.stack(neighbors, dim=1)

    def forward(self, features, xyz, nuv, esm):
        """Inputs: [P,K,12], [P,K,3], unused normals, [P,1280]."""
        xyz = xyz.float()
        features = self.channel_embedding(features.float())
        neighbors = self._subpatch_indices(xyz)
        batch = torch.arange(len(xyz), device=xyz.device)[:, None, None]
        local_features = features[batch, neighbors].flatten(2)
        subpatch_centers = xyz[batch, neighbors].mean(dim=2)
        patch_centers = xyz.mean(dim=1, keepdim=True)
        local = self.to_subpatch_embedding(local_features)
        local = local + self.pos_embedding(subpatch_centers - patch_centers)
        local = self.sub_pool(self.sub_transformer(local).transpose(1, 2)).squeeze(-1)
        centers = patch_centers.squeeze(1).unsqueeze(0)
        positions = centers - centers.mean(dim=1, keepdim=True)
        global_input = local.unsqueeze(0) + self.pos_embedding(positions)
        global_features = self.transformer(global_input)
        esm_features = self.esm_linear(esm.unsqueeze(0))
        # The old in-place position update also affected the residual branch.
        combined = self.ln(torch.cat((global_features + global_input, esm_features), dim=-1))
        return self.predict_head(combined)


def build_model(args):
    return ViT_ESM(depth=args.n_layers, dropout=args.dropout, patch_size=args.num_neighbors)
