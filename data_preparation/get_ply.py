import os 
import sys
import time
import logging
import shutil
import pymesh
import Bio.PDB as PDB
from subprocess import Popen, PIPE
from IPython.core.debugger import set_trace

sys.path.append("/root/vit")
from default_config.dir_opts import dir_opts
from default_config.bin_path import bin_path
from default_config.masif_opts import masif_opts

from data_preparation.parse_input import parse_input
from data_preparation.protonate import protonate
from data_preparation.split_chain import get_chain
from data_preparation.get_fasta import write_fasta
from data_preparation.extractPDB import extractPDB

from data_preparation.triangulation.computeMSMS import computeMSMS
from data_preparation.triangulation.computeCharges import computeCharges, assignChargesToNewMesh
from data_preparation.triangulation.computeAPBS import computeAPBS
from data_preparation.triangulation.computeHydrophobicity import computeHydrophobicity
from data_preparation.triangulation.fixmesh import fix_mesh
from data_preparation.triangulation.compute_normal import compute_normal

from data_preparation.input_output.save_ply import save_ply

def get_ply(pdbid, chainid, rnaids, binding_info, chain_pdb_dir, ply_dir):
    
    chain_pdbfile = os.path.join(chain_pdb_dir, f"{pdbid}_{chainid}.pdb")
    ply_file = os.path.join(ply_dir, f"{pdbid}_{chainid}.ply")

    if not os.path.exists(ply_file):
        # * extract protein chain
        out_filename1 = chain_pdbfile.split(".")[0]
        # * Compute MSMS of surface w/hydrogens

        vertices1, faces1, normals1, names1, areas1 = computeMSMS(chain_pdbfile, protonate=True)
        vertex_hbond = computeCharges(out_filename1, vertices1, names1)
        
        # * For each surface residue, assign the hydrophobicity of its amino acid. 
        vertex_hphobicity = computeHydrophobicity(names1)
        # * If protonate = false, recompute MSMS of surface, but without hydrogens (set radius of hydrogens to 0).
        vertices2 = vertices1
        faces2 = faces1
        # * Fix the mesh.
        mesh = pymesh.form_mesh(vertices2, faces2)
        regular_mesh = fix_mesh(mesh, masif_opts['mesh_res'])
        # * Compute the normals
        vertex_normal = compute_normal(regular_mesh.vertices, regular_mesh.faces)
        # * Assign charges on new vertices based on charges of old vertices (nearest neighbor)
        vertex_hbond = assignChargesToNewMesh(regular_mesh.vertices, vertices1, vertex_hbond, masif_opts)
        vertex_hphobicity = assignChargesToNewMesh(regular_mesh.vertices, vertices1, vertex_hphobicity, masif_opts)
        # * Compute charges
        vertex_charges = computeAPBS(regular_mesh.vertices, out_filename1+".pdb", out_filename1)
        save_ply(ply_file, regular_mesh.vertices,\
                    regular_mesh.faces, normals=vertex_normal, charges=vertex_charges,\
                    normalize_charges=True, hbond=vertex_hbond, hphob=vertex_hphobicity)
    



if __name__ == "__main__":
    print("start")
    if len(sys.argv) <= 1: 
        print("Usage: "+sys.argv[0]+" input dataset (txt file)")
        sys.exit(1)
    
    geobind_file = sys.argv[1]
    set_name = geobind_file.split('.')[0]
    raw_pdb_dir = os.path.join(dir_opts['raw_pdb_dir'], set_name)
    chain_pdb_dir = os.path.join(dir_opts['chain_pdb_dir'], set_name)
    pair_pdb_dir = os.path.join(dir_opts['pair_pdb_dir'], set_name)
    tmp_dir = os.path.join(dir_opts['tmp_dir'], set_name)
    ply_dir = os.path.join(dir_opts['ply_dir'], set_name)

    if not os.path.exists(raw_pdb_dir):
        os.makedirs(raw_pdb_dir)
    
    if not os.path.exists(chain_pdb_dir):
        os.makedirs(chain_pdb_dir)

    if not os.path.exists(pair_pdb_dir):
        os.makedirs(pair_pdb_dir)

    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    if not os.path.exists(ply_dir):
        os.makedirs(ply_dir)

    # *parse file
    # pdbids = parse_pdbid_list(geobind_file)
    input_info = parse_input(geobind_file)
    out_fasta = f"{geobind_file.split('.')[0]}.fasta"
    out_f = open(out_fasta, 'w')
    
    pdblist = PDB.PDBList()
    except_pdbs = []
    success_protein = 0
    print(f"There are total {len(input_info)} proteins.")

    for (_, info) in enumerate(input_info):
        pdbid, chainid, rnaids, binding_info = info
        raw_pdbfile = os.path.join(raw_pdb_dir, pdbid+'.pdb')
        pair_pdbfile = os.path.join(pair_pdb_dir, f"{pdbid}_{chainid}_{''.join(rnaids)}.pdb")
        chain_pdbfile = os.path.join(chain_pdb_dir, f"{pdbid}_{chainid}.pdb")
        ply_file = os.path.join(ply_dir, f"{pdbid}_{chainid}.ply")

        print(f"{pdbid}_{chainid} is on.")
        if not os.path.exists(ply_file):
            # if not os.path.exists(pair_pdbfile):
            #     get_chain(pdbid, chainid, rnaids, raw_pdb_dir, pair_pdb_dir)
            # extractPDB(raw_pdbfile, chain_pdbfile, chainid)
            try:
                time_start = time.time()
                # * get raw pdb files 
                if not os.path.exists(raw_pdbfile):
                    pdb_filename = pdblist.retrieve_pdb_file(pdb_code=pdbid, pdir=raw_pdb_dir, obsolete=False, file_format='pdb')
                    # * do protonation
                    protonate(pdb_filename, raw_pdbfile)
                    os.remove(pdb_filename)
                
                # * get pair pdb files
                if not os.path.exists(pair_pdbfile):
                    get_chain(pdbid, chainid, rnaids, raw_pdb_dir, pair_pdb_dir)
                
                if not os.path.exists(chain_pdbfile):
                    extractPDB(raw_pdbfile, chain_pdbfile, chainid)

                if not os.path.exists(ply_file):
                    get_ply(pdbid, chainid, rnaids, binding_info, chain_pdb_dir, ply_dir)
                time_end = time.time()
                time_use = time_end - time_start
                print(f"{pdbid}_{chainid}: using time {time_use}s, finished.")
            except:
                except_pdbs.append(pdbid)
                continue
        else:
            print(f"{pdbid}_{chainid}: already exists.")
        success_protein += 1

    if len(except_pdbs) != 0:
        print(f"except pdbs: {except_pdbs}")
    else:
        print(f"{success_protein} proteins success.")
    