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

def read_surface(plyfile):

    mesh = pymesh.load_mesh(plyfile)
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
    charge = normalize_electrostatics(charge)
    # Hbond features
    hbond = mesh.get_attribute("vertex_hbond")
    # Hydropathy features
    # Normalize hydropathy by dividing by 4.5
    hphob = mesh.get_attribute("vertex_hphob")/4.5
    chem_props = np.stack([charge, hbond, hphob], axis=1)
    geo_props = np.stack([si, H, K], axis=1)
    geo_props = np.where(np.isnan(geo_props), np.full_like(geo_props, 1e-8), geo_props)
    return coords, faces, normals, chem_props, geo_props

def read_structure(pairfile):
        
    parser = PDBParser()
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
                    resname = AA_CODES[res.get_resname()] # 1-lett
                else:
                    resname = 'X'
                
                seq.append(resname)
                
                for atom in res:
                    atom_type = atom.get_id()[0]
                    if atom_type in self.atom_onehot.keys():
                        atoms_type.append(self.atom_onehot[atom_ty
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