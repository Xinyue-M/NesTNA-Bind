import os 
import sys
import _pickle as cPickle
import numpy as np 
import pandas as pd 
from Bio.PDB import PDBParser, Selection
from sklearn.neighbors import KDTree
import pymesh
import open3d as o3d
# import esm
import torch
import random
sys.path.append("/root/vit")
from utils import *
from default_config.dir_opts import dir_opts
import gdist
from triangulation.fixmesh import fix_mesh

INF = 1e5

def init_esm():
    esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    esm_model.eval()
    return esm_model, batch_converter

class NRB_Processor():

    def __init__(self, pair_anno, taskname, na_type, dir_opts, esm=None, sampling_num=1024, neighbors=128, radius=12):
        
        self.pair =  pair_anno
        print(f"{self.pair} begin processing.")
        self.pdb_id, self.protein_chain, self.binding_chains = self.pair.split('_') # 1ddl:A:D_E
        self.na_type = na_type
        self.taskname = taskname
        self.sampling_num, self.neighbors = sampling_num, neighbors
        self.radius = radius
        self.dir_opts = dir_opts
        self.plyfile = os.path.join(self.dir_opts["ply_dir"], taskname, f"{self.pdb_id}_{self.protein_chain}.ply")
        self.pairfile = os.path.join(self.dir_opts["pair_pdb_dir"], taskname, f"{self.pdb_id}_{self.protein_chain}_{self.binding_chains}.pdb")
        self.pt_file = os.path.join(self.dir_opts['pt_dir'], taskname, f"{self.pdb_id}_{self.protein_chain}.pt")
    
        self.atom_onehot = {"C": 0, "H": 1, "O": 2, "N": 3, "S": 4} # 6-dim
        self.vertices_coord, self.faces, self.normals, self.chem_props, self.geo_props = self._read_surface()
        self.calpha_coord, self.res_ids, self.atoms_coord, self.atoms_type, self.seq = self._read_structure()
        
        try:
            assert len(self.seq) == self.calpha_coord.shape[0]
        except:
            print(len(self.seq), self.calpha_coord.shape[0], "aa")
        # assert len(self.seq) == self.calpha_coord.shape[0]
        assert self.vertices_coord.shape[0] == self.chem_props.shape[0]

        self.vert_resids, self.vert_atomtype = self._get_vert_info() # (N, ) # (N, )
        self.surface_resid, self.surface_idx = self._get_surface_res()  # self.surface_resid[idx] = resid
        self.na_coords = self._get_nucleic_acid_atom() 
        self.interface_lable = self._get_interface(surf_rna_dist=3.0)  # (N, ) 0/1 label
        self.site, self.point_res_idx = self._get_site_biolip()
        
        # self.esm_repr, _, _ = self._esm_process()
        
        self.centroids, self.patches = self._knn()
        # self.esm = self._read_esm(esm)
    

    def get_pair(self):
        return self.pair

    def _read_esm(self, esm):
        centroids_residx = self.point_res_idx[self.centroids].squeeze()
        esm = esm.squeeze()
        patch_esm = esm[centroids_residx]
        # print(patch_esm.shape)
        return patch_esm

    def _read_surface(self):
        
        mesh = pymesh.load_mesh(self.plyfile)
        coords = np.copy(mesh.vertices)
        faces = np.copy(mesh.faces)

        n1 = mesh.get_attribute("vertex_nx")
        n2 = mesh.get_attribute("vertex_ny")
        n3 = mesh.get_attribute("vertex_nz")
        normals = np.stack([n1,n2,n3], axis=1)

        # Compute the principal curvature components for the shape index. 
        mesh.add_attribute("vertex_mean_curvature")
        H = mesh.get_attribute("vertex_mean_curvature")
        mesh.add_attribute("vertex_gaussian_curvature")
        K = mesh.get_attribute("vertex_gaussian_curvature")
        elem = np.square(H) - K
        # In some cases this equation is less than zero, likely due to the method that computes the mean and gaussian curvature.
        # set to an epsilon.
        elem[elem<0] = 1e-8
        k1 = H + np.sqrt(elem)
        k2 = H - np.sqrt(elem)
        # Compute the shape index 
        si = (k1+k2)/(k1-k2)
        si = np.arctan(si)*(2/np.pi) # range(-1, 1)

        # Normalize the charge.
        charge = mesh.get_attribute("vertex_charge")
        # charge = normalize_electrostatics(charge)

        # Hbond features
        hbond = mesh.get_attribute("vertex_hbond")

        # Hydropathy features
        # Normalize hydropathy by dividing by 4.5
        hphob = mesh.get_attribute("vertex_hphob")

        chem_props = np.stack([charge, hbond, hphob], axis=1)
        geo_props = np.stack([si, H, K], axis=1)
        # print(np.count(np.where(np.isnan(geo_props))))
        geo_props = np.where(np.isnan(geo_props), np.full_like(geo_props, 1e-8), geo_props)

        return coords, faces, normals, chem_props, geo_props

    def _read_structure(self):
        
        parser = PDBParser(QUIET=True)
        structrue = parser.get_structure(self.pdb_id, self.pairfile)
        model = Selection.unfold_entities(structrue, "M")[0]

        atoms_coord = []; atoms_type = []; seq = []; res_ids = []; calpha_coord = []
        chains = model.get_chains()
        
        for _, chain in enumerate(chains):
            if chain.id == self.protein_chain:
                residues = list(chain.get_residues())
                for (idx, res) in enumerate(residues):
                    # print(idx, res.get_resname())
                    if res.get_resname() in AA_CODES.keys():
                        resname = AA_CODES[res.get_resname()] # 1-letter
                    else:
                        resname = 'X'
                    
                    seq.append(resname)
                    
                    for atom in res:
                        atom_type = atom.get_id()[0]
                        if atom_type in self.atom_onehot.keys():
                            atoms_type.append(self.atom_onehot[atom_type])
                        else:
                            atoms_type.append(5)
                        atoms_coord.append(atom.coord)
                        # print(atom.get_id()[0])
                    
                    res_id = res.get_id()[1]
                    res_ids.append(res_id)

                    if res.has_id('CA'):
                        calpha = res['CA']
                        calpha_coord.append(calpha.coord)
                    
        calpha_coord = np.asarray(calpha_coord, dtype=np.float64)
        res_ids = np.asarray(res_ids, dtype=np.int32)
        atoms_coord = np.asarray(atoms_coord, dtype=np.float64)
        atoms_type = np.asarray(atoms_type, dtype=np.int32)
        seq = ''.join(seq)

        return calpha_coord, res_ids, atoms_coord, atoms_type, seq

    def _get_vert_info(self):
        
        '''
        To fine the nearest atom and c-alpha for each vertex
        '''
        calpha_kdtree = KDTree(self.calpha_coord)
        calpha_dist, calpha_ind = calpha_kdtree.query(self.vertices_coord) 
        # print(sorted(calpha_dist))
        # print(self.vertices_coord.shape, self.calpha_coord.shape)
        atom_kdtree =KDTree(self.atoms_coord)
        atom_dist, atom_ind = atom_kdtree.query(self.vertices_coord) 
        vert_resids = self.res_ids[calpha_ind]
        # print(calpha_dist)
        vert_atoms = self.atoms_type[atom_ind]
        # return (vert_resids, vert_atoms)
        return vert_resids, vert_atoms

    def _get_site_biolip(self): #主要搞不懂的地方
        '''
        need surface_res, self.vert_info
        surface_res: 在表面上的残基的set
        vert_info: 表面的每个点对应的残基 chain_resid
        '''
        index2surface_res = {}
        surface_res2index = {}
        for index, res_id in enumerate(self.surface_resid):
            index2surface_res[index] = res_id
            surface_res2index[res_id] = index

        point_res_idx = np.zeros_like(self.vert_resids, dtype=np.int32) # (N, )
        for index, point_info in enumerate(self.vert_resids):
            vert_resid = point_info[0]
            point_res_idx[index] = surface_res2index[vert_resid] # idx
        binding_idx = np.unique(point_res_idx[np.where(self.interface_lable == 1)[0]])
        site_label = np.zeros_like(self.surface_resid, dtype=np.int32) # based on index
        site_label[binding_idx] = 1
        # print(site_label)
        # binding_id = np.array(self.surface_resid)[binding_idx]
        # print(binding_id)
        # print(len(self.surface_resid))
        # print(self.surface_resid)
        # anno = anno.split(':')
        # site_by_biolip = set() # id
        # for chain in anno:
        #     chain, binding_residues = chain.split(' ')[0], chain.split(' ')[1:]
        #     chain = chain[0]
        #     for residue in binding_residues:
        #         site_by_biolip.add(int(residue[1:])) # id
        # self.site_by_biolip=site_by_biolip # based on residues
        
        # index2surface_res = {}
        # # index2res_id = []
        # surface_res2index = {}
        # for index, res_id in enumerate(self.surface_resid):
        #     index2surface_res[index] = res_id
        #     surface_res2index[res_id] = index
        #     # index2res_id.append(res_id)
        # residue2label = {surface_res2index[res_id]:[] for res_id in self.surface_resid} 

        # # vert_resids = self.vert_info_new[0]
        # point_res_idx = np.zeros_like(self.vert_resids, dtype=np.int32) # (N, )
        
        # for index, point_info in enumerate(self.vert_resids):
        #     vert_resid = point_info[0]
        #     point_res_idx[index] = surface_res2index[vert_resid] # idx

        #     if vert_resid in site_by_biolip: # 对每个点，如果点最接近的残基gt为1，则残基的点集+1
        #         residue2label[surface_res2index[vert_resid]].append(1)
        #     else:
        #         residue2label[surface_res2index[vert_resid]].append(0)
        # # print(residue2label) # 全0或全1，但是每个残基对应的点的数目不同
        # site_label = np.zeros_like(self.surface_resid, dtype=np.int32) # based on index
        # for index in range(len(self.surface_resid)): # 只考虑表面的残基
        #     site_label[index] = np.max(residue2label[index]) # 如果有1则取1，

        return site_label, point_res_idx # surface res index
        # pass

    def _get_surface_res(self):
        surface_resid = set()
        surface_idx = set()
        resid2index = {}
        index2resid = {}
        for index, res_id in enumerate(self.res_ids):
            index2resid[index] = res_id
            resid2index[res_id] = index
        vert_resids = self.vert_resids.squeeze().tolist()
        for resid in vert_resids:
            surface_resid.add(resid)
            surface_idx.add(resid2index[resid])
        return list(surface_resid), list(surface_idx)

    def _get_nucleic_acid_atom(self):

        if self.na_type == 'RNA':
            ligand_residue_names = ["A", "C", "G", "U"]
        elif self.na_type == 'DNA':
            ligand_residue_names = ["DA", "DC", "DG", "DT"]
        elif self.na_type == 'protein':
            ligand_residue_names = ["ALA", "ARG", "ASN", "ASP", "CYS", 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LYS',' LEU', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', "VAL"]
        na_coords = []
        parser = PDBParser(QUIET=True)
        structrue = parser.get_structure(self.pdb_id, self.pairfile)
        model = Selection.unfold_entities(structrue, "M")[0]
        nucleic_chains = self.binding_chains.split('_')

        for chain_name in nucleic_chains:
            chain = model.child_dict[chain_name]
            
            for res in chain.child_list:
                res_type = res.resname.strip()
                if res_type not in ligand_residue_names:
                    continue
                for atom in res:
                    na_coords.append(atom.coord)
                    # atom_list.append((atom.element, atom.coord, chain.id))
        na_coords = np.asarray(na_coords, dtype=np.float64)

        return na_coords            

    def _get_interface(self, surf_rna_dist):
        
        # NA_coords = np.array([atom[1] for atom in self.nucleic_space])
        kdtree = KDTree(self.na_coords)
        interface = np.zeros([self.vertices_coord.shape[0]], dtype=np.int32)
        for index, vertex in enumerate(self.vertices_coord):
            dis, indice = kdtree.query(vertex[None, :], k=1)
            if dis[0][0] < surf_rna_dist:
                interface[index] = 1
        # print(interface)
        return interface # (N, ) 0/1 label

    def _esm_process(self):
        if self.batch_converter == None or self.esm_model == None:
            return None, None, None
        batch_labels, batch_strs, batch_tokens = self.batch_converter([("prot", self.seq)])
        with torch.no_grad():
            results = self.esm_model(batch_tokens, repr_layers=[33], return_contacts=True)
        token_representations = results["representations"][33][:, 1:-1, :] # (B, L, 1280)
        contact = results["contacts"] # (B, L, L)
        attentions = results["attentions"][0, -1, ...] # (n_layers, n_head, l+2, l+2) -> (n_head, l+2, l+2) 
        print(f"{self.pair} finish esm processing.")
        return token_representations, contact, attentions
        
    def _fps_knn(self):
        '''
        todo: 可以改成根据 geodesic distance 做knn
        '''
        N, M, K = self.vertices_coord.shape[0], self.sampling_num, self.neighbors
        vertices = torch.tensor(self.vertices_coord)
        centroids = torch.zeros((M, ), dtype=torch.long) # index
        distance = torch.ones((N, ), dtype=torch.float64) * INF
        farthest = random.randint(0, N-1)
        patches = torch.zeros((M, K), dtype=torch.long)
        for i in range(M):
            centroids[i] = farthest
            centroid = vertices[farthest][None, :]
            # centroid = vertices[i][None, :]
            dist = torch.sum((vertices - centroid) ** 2, -1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = torch.max(distance, -1)[1]
            _, k_indices = torch.topk(dist, K, largest=False)
            patches[i, :] = k_indices

        return centroids, patches # index (tensor , tensor)


    def _knn(self):

        N = self.vertices_coord.shape[0]
        K = self.neighbors

        vertices = torch.tensor(self.vertices_coord)
        patches = torch.zeros((N, K), dtype=torch.long)
        centroids = torch.zeros((N, ), dtype=torch.long) # index
        for i in range(N):
            centroid = vertices[i][None, :]
            dist = torch.sum((vertices - centroid) ** 2, -1)
            d, k_indices = torch.topk(dist, K, largest=False)
            geo_idx = np.where(d <= self.radius)
            padding_idx = np.where(d > self.radius)
            d = d[geo_idx]
            patches[i, :] = k_indices
            centroids[i] = i
        return centroids, patches

    def _fps_knn_geo(self):
        
        N, M, K = self.vertices_coord.shape[0], self.sampling_num, self.neighbors
        L = self.neighbors
        
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(self.vertices_coord)
        mesh.triangles = o3d.utility.Vector3iVector(self.faces)

        vertices = torch.tensor(self.vertices_coord)
        centroids = torch.zeros((M, ), dtype=torch.long) # index
        distance = torch.ones((N, ), dtype=torch.float64) * INF
        farthest = random.randint(0, N-1)
        patches = torch.zeros((M, L), dtype=torch.long)
        # print(N, M, K)
        for i in range(M):
            centroids[i] = farthest # fixed center
            centroid = vertices[farthest][None, :]
            dist = torch.sum((vertices - centroid) ** 2, -1)
            # print(dist)
            a = gdist.compute_gdist(self.vertices_coord, self.faces, source_indices=np.asarray([farthest], dtype=np.int32), 
                        target_indices=np.asarray([i for i in range(0, N)], dtype=np.int32))
            
            d, k_indices = torch.topk(torch.Tensor(a), L, largest=False)
            
            geo_idx = np.where(d <= self.radius)
            padding_idx = np.where(d > self.radius)
            d = d[geo_idx]
            k_indices[padding_idx] = farthest
            # print(d)
            
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = torch.max(distance, -1)[1]
            patches[i, :] = k_indices

        return centroids, patches # index (tensor , tensor)    

    # def _get_patch(self):

    #     vertices = torch.tensor(self.vertices_coord)[patch_idx]
    #     normals = torch.tensor(self.normals)[patch_idx]
    #     chemprops = torch.tensor(self.chem_props)[patch_idx][None, :, :]
    #     geoprops = torch.tensor(self.geo_props)[patch_idx][None, :, :]

        

    def get_data(self):
        
        nbp_data = {'xyz': self.vertices_coord,  
                    'nuv': self.normals, 
                    'geo': self.geo_props,
                    'chem': self.chem_props,
                    'atom_type': self.vert_atomtype,

                    'interface_label': self.interface_lable,
                    'point2site': self.point_res_idx,

                    'seq': self.seq,
                    # 'esm': self.esm, 
                    'site_label': self.site,
                    'patch_idx': self.patches,
                    'res_idx2id': self.surface_resid,
                    'surface_idx': self.surface_idx
                    }
        # print(self.surface_idx)
        # print(self.seq)
        print(len(self.surface_idx), len(self.seq))
        save_path = os.path.join(self.dir_opts['pt_dir'], self.taskname, self.pair)
        with open(save_path, 'wb') as pid:
            cPickle.dump(nbp_data, pid)
        print(f"{self.pair} finish dumping.\n")
        return



if __name__ == "__main__":

    print("start")
    # taskname = "test"
    # taskname = "RNA-157_Test"
    # taskname = "RNA-663_Train"
    # taskname = "DNA-179_Test"
    # taskname = "DNA-719_Train"
    taskname = sys.argv[1]
    na_type = sys.argv[2]
    sampling_num, neighbors, radius = sys.argv[3], sys.argv[4], sys.argv[5]
    # dir_opts['pt_dir'] = f'data/pt_{sampling_num}_{neighbors}_{radius}A'
    dir_opts['pt_dir'] = f'data/dmasif'

    print(sampling_num, neighbors)
    anno_file = f"{taskname}.txt"

    with open(anno_file, 'r') as f:
        lines = f.read().split("\n")
    f.close()

    if not os.path.exists(os.path.join(dir_opts['pt_dir'], taskname)):
        os.makedirs(os.path.join(dir_opts['pt_dir'], taskname))

    except_case = []
    # esm_model, batch_converter = init_esm()
    # esm_model = None; batch_converter = None
    # esm_file = os.path.join(dir_opts['esm_dir'], taskname)
    # print(esm_file)
    # if os.path.exists(esm_file):
    #     add_esm = True
    #     with open(esm_file, 'rb') as f:
    #         esm_info = cPickle.load(f)
    #     f.close()
    # print(len(esm_info.keys()))
    # print(lines)
    for line in lines[:]:
        pair = line.split("\t")[0]
        print(pair)
        if not os.path.exists(os.path.join(dir_opts['pt_dir'], taskname, pair)):
            if line.startswith("3iem:A:G_J_L") or line.startswith("1vq8:Q:0_9"):
                continue
            # nbp = NRB_Processor(line, taskname, na_type, dir_opts, sampling_num=int(sampling_num), neighbors=int(neighbors))
            # nbp.get_data()
            try:
                # esm = esm_info['_'.join(pair.split(':'))][-1]
                esm = None
                # print(esm.shape)
                nbp = NRB_Processor(line, taskname, na_type, dir_opts, sampling_num=int(sampling_num), neighbors=int(neighbors), radius=int(radius))
                nbp.get_data()
            except:
                except_case.append(pair)
        else:
            print(f"{pair} already exsits.")
            pass
        # break
    print(except_case)