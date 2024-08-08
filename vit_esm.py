import torch
from torch import nn
import numpy as np

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import random
# from pointnet2_utils import PointNetSetAbstraction
# from pointmlp import PointMLPEocoder
# helpers
INF = 1e5

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def pc_normalize(pc, centroid):
    '''
    pc: (B, N, 3))
    centroid: (B, 1, 3)
    '''
    pc = pc - centroid
    m = torch.max(torch.sqrt(torch.sum(pc **2, dim=-1)))
    pc = pc / m
    return pc

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, input_dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)

        inner_dim = dim_head *  heads # 512
        project_out = not (heads == 1 and dim_head == input_dim)
        
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(input_dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, input_dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        # print(f"attn input.shape {x.shape}")
        x = self.norm(x) # layernorm first

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)') # concat
        return self.to_out(out)

  
      
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)

class ViT_ESM(nn.Module):
    def __init__(self, num_classes, dim, depth, heads, mlp_dim, patch_dim=256, hidden_dim=256, esm_dim=256, patch_size=256, num_patches=1024, pool = 'mean', channels = 12, dim_head = 64, dropout = 0.1):
        super().__init__()
        patch_input_dim = channels * patch_size

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        # assert esm_dim == patch_dim == dim
        patch_dim = dim
        esm_dim = dim
        assert esm_dim == patch_dim == dim
        # self.point_encoder = pointMLPElite()

        self.channel_embedding = nn.Sequential(
            nn.Linear(channels, 32),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(32, 32),
            nn.LeakyReLU(negative_slope=0.2)
        )
        # * add a norm
        self.norm_in = nn.LayerNorm(128)

        
        self.sub_neighbor = 16
        self.sub_patch = 16
        self.to_subpatch_embedding = nn.Sequential(
            nn.LayerNorm(self.sub_neighbor * 32),
            nn.Linear(self.sub_neighbor * 32, patch_dim),
            nn.LayerNorm(patch_dim),
            # nn.ReLU(),
            # nn.Linear(patch_dim, patch_dim),
            # nn.Dropout(p=dropout)
        ) # M*D -> 256

        self.sub_pool = nn.MaxPool1d(self.sub_patch, self.sub_patch)
        self.sub_transformer = Transformer(dim, depth, heads, 8, mlp_dim, dropout)

        self.esm_linear = nn.Linear(1280, esm_dim) 

        self.pos_embedding = nn.Sequential(
            nn.Linear(3, patch_dim).to(torch.float32),
            nn.GELU(),
            nn.Linear(patch_dim, patch_dim).to(torch.float32),
            # nn.Dropout(p=dropout)
        )
        
        
        self.dropout = nn.Dropout(dropout)

        self.transformer = Transformer(dim * 2, depth, heads, 8, mlp_dim, dropout)

        self.ln = nn.LayerNorm(dim * 2)

        # self.pool = pool
        self.pool = torch.nn.MaxPool1d(2, 2)
        self.to_latent = nn.Identity()

        self.predict_head = nn.Sequential(
            nn.Linear(dim, 16),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(16, 16),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(16, 2)
        )

    def forward(self, features, xyz, nuv, esm):
        # patches: (B, N, patch_size, C) (1, 1024, 256, 12)
        # xyz: (B, N, patch_size, 3)
        # esm_embedding: (B, N, 1280) N = 1024
        # features = features.unsqueeze(dim=0)
        # xyz = xyz.unsqueeze(dim=0)
        # nuv = nuv.unsqueeze(dim=0)
        esm = esm.unsqueeze(dim=0)
     

        N, M, D = features.size()
        xyz = xyz.to(torch.float32)
        features = features.to(torch.float32)
        features = self.channel_embedding(features) # (N, M, C) -> (N, M, D)
        # print(features.shape)

        # * local patch learning 
        # centroids, patches = self._fps_knn_ori(xyz, sub_neighbor=self.sub_neighbor, sub_patch=self.sub_patch) # patches: (N, self.sub_neighbor, self.sub_patch)
        centroids, patches = self._fps_knn(xyz, sub_neighbor=self.sub_neighbor, sub_patch=self.sub_patch, radius=20) # patches: (N, self.sub_neighbor, self.sub_patch)
        patches_reshape = patches.view(-1, self.sub_patch * self.sub_neighbor) # patches_reshape: (N, self.sub_neighbor * self.sub_patch)
        # print(self.sub_patch * self.sub_neighbor)
        idx = torch.tensor([i for i in range(N)]).unsqueeze(1).repeat(1, self.sub_patch * self.sub_neighbor)
        features_inpatch = features[idx, patches_reshape].reshape(N, self.sub_patch, self.sub_neighbor, -1) # features_inpatch: (N, self.sub_neighbor, self.sub_patch, D)
        # features_inpatch = self.channel_embedding(features_inpatch) # (N, M, C) -> (N, M, D)
        features_inpatch = features_inpatch.reshape(N, self.sub_patch, -1).to(torch.float32) # features_inpatch: (N, self.sub_neighbor, self.sub_patch * D)
        xyz_inpatch = xyz[idx, patches_reshape].reshape(N, self.sub_patch, self.sub_neighbor, -1)
        subpatch_centers = torch.mean(xyz_inpatch, dim=2, keepdim=False) # center_inpatch (N, self.sub_patch, 3)
        # print(subpatch_centers)
        patch_centers = torch.mean(xyz, dim=1, keepdim=True) # (N, 1, 3)
        # subpatch_centers = torch.cat((patch_centers, subpatch_centers), dim=1) # (N, self.sub_patch+1, 3)
        subpatch_centers -= patch_centers
        # print(subpatch_centers)
        # subpatch_cls_token = repeat(self.cls_token, '1 1 d -> n 1 d', n = N)
        # print(features_inpatch.shape)
        # print(subpatch_embedding.shape)
        subpatch_embedding = self.to_subpatch_embedding(features_inpatch)  # subpatch_embedding: (N, self.sub_neighbor, D')
        # subpatch_embedding = torch.cat((subpatch_cls_token, subpatch_embedding), dim=1)  # subpatch_embedding: (N, self.sub_neighbor + 1, D')
        subpatch_embedding += self.pos_embedding(subpatch_centers)
        # subpatch_embedding = self.dropout(subpatch_embedding)
        subpatch_embedding = self.sub_transformer(subpatch_embedding)
        subpatch_embedding = self.sub_pool(subpatch_embedding.permute(0, 2, 1)).squeeze() # (N, D')
        # subpatch_embedding = subpatch_embedding[:, 0, :].squeeze()

        # print(subpatch_embedding.shape)
        
        # * global patch learning
        patch_embedding = subpatch_embedding.unsqueeze(dim=0) # (B, N, D')
        B = patch_embedding.shape[0]
        xyz = xyz.unsqueeze(dim=0)
        patch_centers = patch_centers.squeeze()[None, ...] # (B, N, 3)
        patch_center = torch.mean(patch_centers, dim=1)
        # print(patch_center)
        patch_centers -= patch_center
        # print(patch_centers)
        # cls_center = torch.mean(patch_centers, dim=1, keepdim=True) # (B, 1, 3)
        # patch_centers = torch.cat((cls_center, patch_centers), dim=1) # (B, N+1, 3)
        esm_embedding = self.esm_linear(esm) # (B, N, 1280) -> (B, N, dim')
        # esm_embedding_cls = repeat(self.esm_cls_token, '1 1 d -> b 1 d', b = B) # (B, 1, 1280) # todo: change esm_embedding to (B, L, 1280)
        # esm_embedding = torch.cat((esm_embedding_cls, esm_embedding), dim=1) # (B, N+1, dim)

        x = patch_embedding
        # cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = B)
        # x = torch.cat((cls_tokens, patch_embedding), dim=1) # (B, N+1, dim)
        # x = torch.cat((x, esm_embedding), dim=-1)
        
        # position embedding 
        x += self.pos_embedding(patch_centers) # (B, N+1, dim)
        
        # x = self.dropout(x) # (B, N+1, dim)
        x = self.ln(torch.cat((x, esm_embedding), dim=-1))
        x = self.dropout(x) # (B, N+1, dim)

        # cross attention decoder 
        self_x = self.transformer(x) # (B, N+1, dim)
        # output = self_x
        output = self.pool(self_x) + subpatch_embedding
        return self.predict_head(output)
    
    def _patch2site(self, x, patch_in_site, n_patch2site):
        
        S, D = n_patch2site.size()[0], x.size()[-1]
        x = x.squeeze()
        site_x = torch.zeros((S, D)).to(next(self.parameters()).device)
        for i in range(S):
            site_x[i, :] = torch.mean(x[patch_in_site[n_patch2site[i]]], dim=0)
        
        return site_x.unsqueeze(0)

    def _fps_knn(self, xyz, sub_patch, sub_neighbor, radius):
        '''
        xyz: (B, 256, 12)
        # B: batch number
        # N: total number of sampling
        # M: patch number
        # K: neighbors in patch
        '''
        B, N, M, K = xyz.shape[0], xyz.shape[1], sub_patch, sub_neighbor
        vertices = xyz
        dist = torch.sqrt(torch.sum((xyz - xyz[:, 0, :][:, None, :]) ** 2, -1))
        mask = torch.zeros((xyz.shape[0], xyz.shape[1]))
        mask[dist > radius] = -1

        centroids = torch.zeros((B, M, ), dtype=torch.long) # index
        distance = (torch.ones((B, N, ), dtype=torch.float32) * INF).to(next(self.parameters()).device)
        distance[mask == -1] = -INF

        mask = mask.unsqueeze(2).repeat(1, 1, 3)
        vertices[mask == -1] = 1e5

        farthest = torch.randint(0, 50, (B, )).to(next(self.parameters()).device)
        patches = torch.zeros((B, M, K), dtype=torch.long).to(next(self.parameters()).device)
        for i in range(M):
            centroids[:, i] = farthest
            centroid = vertices[[j for j in range(B)], farthest]
            # centroid = vertices[i][None, :]
            dist = torch.sum((vertices - centroid[:, None, :]) ** 2, -1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = torch.max(distance, -1)[1]
            _, k_indices = torch.topk(dist, K, largest=False)
            patches[:, i] = k_indices

        return centroids, patches # index (tensor , tensor)

    def _fps_knn_ori(self, xyz, sub_patch, sub_neighbor):
        '''
        xyz: (B, 256, 12)
        # B: batch number
        # N: total number of sampling
        # M: patch number
        # K: neighbors in patch
        '''
        B, N, M, K = xyz.shape[0], xyz.shape[1], sub_patch, sub_neighbor
        vertices = xyz
        centroids = torch.zeros((B, M, ), dtype=torch.long) # index
        distance = (torch.ones((B, N, ), dtype=torch.float32) * INF).to(next(self.parameters()).device)
        farthest = torch.randint(0, N-1, (B, )).to(next(self.parameters()).device)
        patches = torch.zeros((B, M, K), dtype=torch.long).to(next(self.parameters()).device)
        for i in range(M):
            centroids[:, i] = farthest
            centroid = vertices[[j for j in range(B)], farthest]
            # centroid = vertices[i][None, :]
            dist = torch.sum((vertices - centroid[:, None, :]) ** 2, -1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = torch.max(distance, -1)[1]
            _, k_indices = torch.topk(dist, K, largest=False)
            patches[:, i] = k_indices

        return centroids, patches # index (tensor , tensor)
