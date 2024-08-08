import os 
import sys
import time
import logging
import shutil
import pymesh
import Bio.PDB as PDB
import pymeshfix
from subprocess import Popen, PIPE
from IPython.core.debugger import set_trace

sys.path.append("/root/vit")
from default_config.dir_opts import dir_opts
from default_config.bin_path import bin_path
from default_config.masif_opts import masif_opts

from data_preparation.parse_input import parse_input, parse_dmasif
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

def fix_ply(vertices0, faces0, normals0, names0):

    # print("vertices:", len(vertices0)) 
    # print("faces:", len(faces0))

    normals1 = []; names1 = []
    # meshfix = pymeshfix.MeshFix(vertices0, faces0)
    # meshfix.repair()
    # meshfix.write('out.ply')
    vertices1, faces1 = pymeshfix.clean_from_arrays(vertices0, faces0)
    ptr1 = 0
    # print("vertices1:", len(vertices1)) 
    # print("faces1:", len(faces1))
    # print(vertices0)
    # print(vertices1)
    for i in range(len(vertices1)):
        if vertices0[i].all() == vertices1[ptr1].all():
            print(vertices0[i], vertices1[ptr1])
            normals1.append(normals0[i])
            names1.append(names0[i])
            ptr1 += 1
            # print(ptr1)
    return vertices1, faces1, normals1, names1


if __name__ == "__main__":
    print("start")
    if len(sys.argv) <= 1: 
        print("Usage: "+sys.argv[0]+" input dataset (txt file)")
        sys.exit(1)
    
    input = sys.argv[1]
    set_name = input.split('.')[0]
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
    input_info = parse_input(input)
    # input_info = parse_dmasif(input)
    print(len(input_info))
    # * =======end========
    out_fasta = f"{input.split('.')[0]}.fasta"
    out_f = open(out_fasta, 'w')
    
    pdblist = PDB.PDBList()
    except_pdbs = []
    success_protein = 0

    for (_, info) in enumerate(input_info[700:]):
        pdbid, chainid, rnaids, binding_info = info
        raw_pdbfile = os.path.join(raw_pdb_dir, pdbid+'.pdb')
        pair_pdbfile = os.path.join(pair_pdb_dir, f"{pdbid}_{chainid}_{''.join(rnaids)}.pdb")
        ply_file = os.path.join(ply_dir, f"{pdbid}_{chainid}.ply")
        # write_fasta(pdbid, chainid, rnaids, binding_info, chain_pdbfile, out_f)
        # try:
        print(f"{pdbid}_{chainid} is on.")
        # write_fasta(pdbid, chainid, rnaids, binding_info, chain_pdbfile, out_f)
        # continue
        time_start = time.time()
        # * get rawpdb files
        try:
            if not os.path.exists(raw_pdbfile):
                pdb_filename = pdblist.retrieve_pdb_file(pdb_code=pdbid, pdir=raw_pdb_dir, file_format="pdb", obsolete=False)
                # * do protonation
                protonate(pdb_filename, raw_pdbfile)
                os.remove(pdb_filename)
            # * get pair files
            if not os.path.exists(pair_pdbfile):
                extractPDB(raw_pdbfile, pair_pdbfile, f"{chainid}{rnaids}")
                # get_chain(pdbid, chainid, rnaids, raw_pdb_dir, pair_pdb_dir)
            # * write fasta
            # write_fasta(pdbid, chainid, rnaids, binding_info, chain_pdbfile, out_f)
            # continue
            if not os.path.exists(ply_file):
                # print(ply_file)
                # * extract protein chain
                chain_pdbname = chain_pdb_dir+"/"+pdbid+"_"+chainid
                if not os.path.exists(chain_pdbname+".pdb"):
                    extractPDB(raw_pdbfile, chain_pdbname+".pdb", chainid)
                # * Compute MSMS of surface w/hydrogens
                # vertices: list of coords
                # faces: list of triangles
                # normals: list of normals
                vertices1, faces1, normals1, names1, areas1 = computeMSMS(chain_pdbname+".pdb", protonate=True)
  
                # * compute charge  
                vertex_hbond = computeCharges(chain_pdbname, vertices1, names1)
                
                # * For each surface residue, assign the hydrophobicity of its amino acid. 
                vertex_hphobicity = computeHydrophobicity(names1)
                # * If protonate = false, recompute MSMS of surface, but without hydrogens (set radius of hydrogens to 0).
                vertices2 = vertices1
                faces2 = faces1
                # print(vertex_hbond.shape, vertex_hphobicity.shape)
                # * Fix the mesh.
                mesh = pymesh.form_mesh(vertices2, faces2)
                regular_mesh = fix_mesh(mesh, 1.2)
                # test = fix_mesh(mesh, 1.0)
                print(vertices2.shape[0], regular_mesh.vertices.shape[0])
                # * Compute the normals
                vertex_normal = compute_normal(regular_mesh.vertices, regular_mesh.faces)
                # * Assign charges on new vertices based on charges of old vertices (nearest neighbor)
                vertex_hbond = assignChargesToNewMesh(regular_mesh.vertices, vertices1, vertex_hbond, masif_opts)
                vertex_hphobicity = assignChargesToNewMesh(regular_mesh.vertices, vertices1, vertex_hphobicity, masif_opts)
                
                
                # * Compute charges
                vertex_charges = computeAPBS(regular_mesh.vertices, chain_pdbname+".pdb", chain_pdbname)
                save_ply(ply_file, regular_mesh.vertices,\
                            regular_mesh.faces, normals=vertex_normal, charges=vertex_charges,\
                            normalize_charges=True, hbond=vertex_hbond, hphob=vertex_hphobicity)
                time_end = time.time()
                time_use = time_end - time_start
                
                print(f"{pdbid}_{chainid}: using time {time_use}s, finished.")
            else:
                # print(f"{pdbid}_{chainid}: already exists.")
                pass
            success_protein += 1
            
        except:
            # set_trace()
            except_pdbs.append(f"{pdbid}_{chainid}")
            continue
        
        # break
    if len(except_pdbs) != 0:
        print(f"except pdbs: {except_pdbs}")
    else:
        print(f"{success_protein} proteins success.")
    

    # write fasta
    # out_fasta = f"{geobind_file.split('.')[0]}.fasta"
    # with open(out_fasta, 'w') as f:
    #     pass 
    # parse_label(geobind_file, dir_opts['chain_pdb_dir'], f"{geobind_file.split('.')[0]}.fasta")