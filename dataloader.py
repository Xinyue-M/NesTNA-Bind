import os, sys
import pickle
import _pickle as cPickle
from typing import Iterator
import torch 
from torch.utils.data import Dataset, DataLoader, IterableDataset
import torch.nn.functional as F

import esm

sys.path.append("/root/vit")

def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    batchsize, ndataset, dimension = xyz.shape
    centroids = torch.zeros(batchsize, npoint, dtype=torch.long).to(device)
    distance = torch.ones(batchsize, ndataset).to(device) * 1e10
    farthest =  torch.randint(0, ndataset, (batchsize,), dtype=torch.long).to(device)
    batch_indices = torch.arange(batchsize, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:,i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(batchsize, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def bincount_2d(pointsite, site_label):

    pointsite = pointsite.squeeze() # (B, N, 1) -> (B, N)
    B, N = pointsite.size()
    S = site_label.size()[0]
    patch_label = torch.zeros((B, ), dtype=torch.long)
    patch_mainidx = torch.zeros((B, ), dtype=torch.long)
    patch_residx_dict = {k:[] for k in range(B)}
    residx_patch_dict = {k:[] for k in range(S)}
    for i in range(B):
        all_idx = torch.where(torch.bincount(pointsite[i]) != 0)[0]
        # print(all_idx)
        # print(S, torch.bincount(pointsite[i]).shape)
        main_res_idx = torch.argmax(torch.bincount(pointsite[i])) # 要修改，每个patch对应多个residue
        patch_mainidx[i] = main_res_idx
        patch_label[i] = site_label[main_res_idx]
        patch_residx_dict[i] = all_idx
        for idx in all_idx:
            residx_patch_dict[idx.item()].append(i)
    # print(residx_patch_dict)
    return patch_label, patch_mainidx, patch_residx_dict, residx_patch_dict
    

class IterPatchDataset(IterableDataset):

    def __init__(self, file_dir, set_type, esm_info=None, file_list=None) -> None:
        super().__init__()
        self.file_dir = file_dir
        self.esm_info = esm_info
        self.set_type = set_type
        
        # self.esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        if file_list == None:
            self.file_list = os.listdir(file_dir)[:]
        else: 
            self.file_list = file_list[:]
        # self._init_esm()
        self.file_list = self.file_list[:]

    def __iter__(self) -> Iterator:
        data = self.process()
        return data   
    
    def _get_patch_label(self, point2site, site_label, patch_idx):

        patch_pointsite = point2site[patch_idx].squeeze()
        B, S = patch_pointsite.size()[0], site_label.size()[0]
        patch_label = torch.zeros((B, ), dtype=torch.long)
        patch_mainidx = torch.zeros((B, ), dtype=torch.long) 
        for i in range(B):
            all_idx = torch.where(torch.bincount(patch_pointsite[i]) != 0)[0]
            main_res_idx = torch.argmax(torch.bincount(patch_pointsite[i])) # 要修改，每个patch对应多个residue
            patch_mainidx[i] = main_res_idx
            patch_label[i] = site_label[main_res_idx]

        return patch_label, patch_mainidx

    def process(self):

        for i, file in enumerate(self.file_list):
            pdbid = file.split('.')[0]
            # print(file)
            with open(os.path.join(self.file_dir, file), 'rb') as f:
                data = cPickle.load(f)
                
                seq = data['seq'] # list (len = protein_len)
                site_label = torch.tensor(data['site_label'], dtype=torch.long)
                y = torch.tensor(data['interface_label'], dtype=torch.long)
                point2site = torch.tensor(data['point2site'], dtype=torch.long)
                res_idx2id = torch.tensor(data['res_idx2id'], dtype=torch.long)
                xyz = torch.tensor(data['xyz'], dtype=torch.float)
                patch_idx = torch.tensor(data['patch_idx'], dtype=torch.long)

                if self.set_type == "train":
                    if xyz.shape[0] > 2048:
                        sampled_v = farthest_point_sample(xyz.unsqueeze(0), 2048)
                        patch_idx = patch_idx[sampled_v.squeeze()]
                else: # valid or test
                    if xyz.shape[0] > 8000:
                        sampled_v = farthest_point_sample(xyz.unsqueeze(0), 8000)
                        patch_idx = patch_idx[sampled_v.squeeze()]
                # print(patch_idx.shape)
                xyz = torch.tensor(data['xyz'])[patch_idx]
                nuv = torch.tensor(data['nuv'])[patch_idx]
                geo = torch.tensor(data['geo'])[patch_idx]
                chem = torch.tensor(data['chem'])[patch_idx]
                atomtype = torch.tensor(data['atom_type'])[patch_idx]
                atomtype = F.one_hot(atomtype.squeeze().type(torch.long), num_classes=6)
                # esm = data['esm']
                patch_label, patch_mainidx = self._get_patch_label(point2site, site_label, patch_idx)
                patch2site = patch_mainidx
                centroid2site = point2site[patch_idx[:, 0].squeeze()].squeeze()
                try:
                    esm = self.esm_info[file][-1].squeeze()[centroid2site]
                except:
                    esm = None
                    print(pdbid)
                
                features = torch.cat((chem, geo, atomtype), dim=-1)
                if y.shape[0] != point2site.shape[0]:
                    print(pdbid)
                yield (pdbid, features, xyz, nuv, esm, patch_label, y, site_label, point2site, patch_idx, patch2site)

    def _init_esm(self):
        self.esm_model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.batch_converter = self.alphabet.get_batch_converter()
        self.esm_model.eval()

    def _esm_process(self, seq, pdbid):

        seq_str = ''.join(seq)
        # if os.path.exists(os.path.join(self.esm_dir, f"{pdbid}.pt")):
        #     token_representations, contact, attentions = torch.load(os.path.join(self.esm_dir, f"{pdbid}.pt"))
        # else:
        #     batch_labels, batch_strs, batch_tokens = self.batch_converter([("prot", seq_str)])
        #     with torch.no_grad():
        #         results = self.esm_model(batch_tokens, repr_layers=[33], return_contacts=True)
        #     token_representations = results["representations"][33][:, 1:-1, :] # (B, L, 1280)
        #     contact = results["contacts"] # (B, L, L)
        #     attentions = results["attentions"][0, -1, ...] # (n_layers, n_head, l+2, l+2) -> (n_head, l+2, l+2)            
        #     torch.save((token_representations, contact, attentions), os.path.join(self.esm_dir, f"{pdbid}.pt"))
        batch_labels, batch_strs, batch_tokens = self.batch_converter([("prot", seq_str)])
        with torch.no_grad():
            results = self.esm_model(batch_tokens, repr_layers=[33], return_contacts=True)
        token_representations = results["representations"][33][:, 1:-1, :] # (B, L, 1280)
        contact = results["contacts"] # (B, L, L)
        attentions = results["attentions"][0, -1, ...] # (n_layers, n_head, l+2, l+2) -> (n_head, l+2, l+2)            
        # torch.save((token_representations, contact, attentions), os.path.join(self.esm_dir, f"{pdbid}.pt"))
                
        return token_representations, contact, attentions
        
    def _get_patch_site(self, patch_in_site, max_patch=8):
        
        patch2site = []
        n_patch2site = []
        for k, v in patch_in_site.items():
            n_patch2site.append(min(len(v), 8))
            if len(v) > max_patch:
                v = v[:max_patch]
            else:
                v = F.pad(v, (0, 8 - len(v)), "constant", -1)
            patch2site.append(v)
        patch2site = torch.stack(patch2site, dim=0)
        n_patch2site = torch.tensor(n_patch2site)
        # print(n_patch2site.shape, patch2site.shape)
        # print(n_patch2site)
        return patch2site, n_patch2site
        pass

    def __iter__(self) -> Iterator:
        data = self.process()
        return data

    def __len__(self):
        return len(self.file_list)

class PatchDataset(Dataset):

    def __init__(self, file_dir, esm_info=None, file_list=None) -> None:
        super().__init__()
        self.file_dir = file_dir
        self.esm_info = esm_info
        
        # self.esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        if file_list == None:
            self.file_list = os.listdir(file_dir)
        else: 
            self.file_list = file_list
        # self._init_esm()
        self.file_list = self.file_list[:]
        self.features, self.coords, self.nuvs, self.esms, self.patch_labels, self.patch_indices = self.process()

    def _get_patch_label(self, point2site, site_label, patch_idx):

        patch_pointsite = point2site[patch_idx].squeeze()
        B, S = patch_pointsite.size()[0], site_label.size()[0]
        patch_label = torch.zeros((B, ), dtype=torch.long)
        patch_mainidx = torch.zeros((B, ), dtype=torch.long) 
        for i in range(B):
            all_idx = torch.where(torch.bincount(patch_pointsite[i]) != 0)[0]
            main_res_idx = torch.argmax(torch.bincount(patch_pointsite[i])) # 要修改，每个patch对应多个residue
            patch_mainidx[i] = main_res_idx
            patch_label[i] = site_label[main_res_idx]

        return patch_label, patch_mainidx

    def process(self):
        
        features, coords, nuvs, esms, patch_labels, ys, sitelabels, point2sites, patch_indices = [], [], [], [], [], [], [], [], []

        for i, file in enumerate(self.file_list):
            pdbid = file.split('.')[0]
            with open(os.path.join(self.file_dir, file), 'rb') as f:
                data = cPickle.load(f)
                
                seq = data['seq'] # list (len = protein_len)
                site_label = torch.tensor(data['site_label'], dtype=torch.long)
                y = torch.tensor(data['interface_label'], dtype=torch.long)
                point2site = torch.tensor(data['point2site'], dtype=torch.long)
                res_idx2id = torch.tensor(data['res_idx2id'], dtype=torch.long)
                patch_idx = torch.tensor(data['patch_idx'], dtype=torch.long)
                xyz = torch.tensor(data['xyz'])[patch_idx]
                nuv = torch.tensor(data['nuv'])[patch_idx]
                geo = torch.tensor(data['geo'])[patch_idx]
                chem = torch.tensor(data['chem'])[patch_idx]
                atomtype = torch.tensor(data['atom_type'])[patch_idx]
                atomtype = F.one_hot(atomtype.squeeze().type(torch.long), num_classes=6)
                centroids = patch_idx[:, 0].squeeze()
                centroids_label = y[centroids]
                centroids_point2site = point2site[centroids].squeeze()
                # esm = data['esm']
                patch_label, patch_mainidx = self._get_patch_label(point2site, site_label, patch_idx)
                
                
                
                try:
                    esm = self.esm_info[file][-1].squeeze()[patch_mainidx]
                except:
                    continue
                feat = torch.cat((chem, geo, atomtype), dim=-1)
                features.append(feat.unsqueeze(dim=0))
                coords.append(xyz.unsqueeze(dim=0))
                nuvs.append(nuv.unsqueeze(dim=0))
                esms.append(esm.unsqueeze(dim=0))
                patch_labels.append(patch_label.unsqueeze(dim=0))
                # ys.append(y.unsqueeze(dim=0))
                # print(y.shape)
                # sitelabels.append(site_label.unsqueeze(dim=0))
                # point2sites.append(point2site.unsqueeze(dim=0))
                patch_indices.append(patch_idx.unsqueeze(dim=0))
                f.close()
                # yield (features, xyz, nuv, esm, patch_label, y, site_label, point2site, patch_idx)
        return torch.cat(features, dim=0), \
               torch.cat(coords, dim=0), \
               torch.cat(nuvs, dim=0), \
               torch.cat(esms, dim=0), \
               torch.cat(patch_labels, dim=0), \
               torch.cat(patch_indices, dim=0)

    def __getitem__(self, index):
        return self.features[index, ...], \
               self.coords[index, ...], \
               self.nuvs[index, ...], \
               self.esms[index, ...], \
               self.patch_labels[index, ...], \
               self.patch_indices[index, ...]
    
    def __len__(self):

        return self.coords.shape[0]
    

if __name__ == "__main__":
    
    print("start")
    taskname = "RNA-157_Test"
    train_dir = "./pt_2048_128_esm/RNA-663_Train"
    test_dir = "./pt_2048_128_esm/RNA-157_Test"
    test_esmfile = './esm/RNA-157_Test'
    if os.path.exists(test_esmfile):
        add_esm = True
        with open(test_esmfile, 'rb') as f:
            esm_info = cPickle.load(f)
        f.close()
    # train_esmdir = "pdb\RNA_Train_surface\esm"
    # test_esmdir = "pdb\RNA_Test_surface\esm"
    # for dir in [train_esmdir, test_esmdir]:
    #     if not os.path.exists(dir):
    #         os.makedirs(dir)

    # train_proteins = os.listdir(train_dir)
    file = "1b2m_A_C"
    esm_info = esm_info[file]
    print(len(esm_info))

    # dataset = IterPatchDataset(test_dir, esm_info)
    # # protein_dl = DataLoader(dataset, shuffle=False)

    # for i, ds in enumerate(dataset):

    #     site = ds[0].squeeze()# one-hot
    #     break
