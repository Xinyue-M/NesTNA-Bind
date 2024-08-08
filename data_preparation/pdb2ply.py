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
if __name__ == "__main__":
    print("start")
    # if len(sys.argv) <= 1: 
    #     print("Usage: "+sys.argv[0]+" input dataset (txt file)")
    #     sys.exit(1)
    
    # geobind_file = sys.argv[1]
    # set_name = geobind_file.split('.')[0]
    # raw_pdb_dir = os.path.join(dir_opts['raw_pdb_dir'], set_name)
    chain_pdb_dir = './data/seq82'
    # pair_pdb_dir = os.path.join(dir_opts['pair_pdb_dir'], set_name)
    # tmp_dir = os.path.join(dir_opts['tmp_dir'], set_name)
    ply_dir = './data/seq82'


    # *parse file
    # pdbids = parse_pdbid_list(geobind_file)
    # input_info = parse_input(geobind_file)
    # out_fasta = f"{geobind_file.split('.')[0]}.fasta"
    # out_f = open(out_fasta, 'w')
    
    # pdblist = PDB.PDBList()
    except_pdbs = []
    success_protein = 0
    pdbs = os.listdir("./data/seq82")
    for pdb in pdbs:
        # pdbid, chainid, rnaids, binding_info = info
        # print(pdbid, chainid, rnaids, binding_info)
        # raw_pdbfile = os.path.join(raw_pdb_dir, pdbid+'.pdb')
        # pair_pdbfile = os.path.join(pair_pdb_dir, f"{pdbid}_{chainid}_{''.join(rnaids)}.pdb")
        pdb_file = os.path.join(chain_pdb_dir, pdb)
        ply_file = os.path.join(ply_dir, f"{pdb[:-4]}.ply")
        # write_fasta(pdbid, chainid, rnaids, binding_info, chain_pdbfile, out_f)
        try:
            # print(f"{pdbid}_{chainid} is on.")
            # write_fasta(pdbid, chainid, rnaids, binding_info, chain_pdbfile, out_f)
            # continue
            time_start = time.time()
            # * get rawpdb files
            # if not os.path.exists(raw_pdbfile):
            #     pdb_filename = pdblist.retrieve_pdb_file(pdb_code=pdbid, pdir=raw_pdb_dir, obsolete=False)
            #     # * do protonation
            #     protonate(pdb_filename, raw_pdbfile)
            #     os.remove(pdb_filename)

            # * get pair files
            # if not os.path.exists(pair_pdbfile):
            #     get_chain(pdbid, chainid, rnaids, raw_pdb_dir, pair_pdb_dir)

            # * write fasta
            # write_fasta(pdbid, chainid, rnaids, binding_info, chain_pdbfile, out_f)
            # continue
            if not os.path.exists(ply_file):
                # * extract protein chain
                chain_pdbname = chain_pdb_dir+"/"+pdb[:-4]
                # if not os.path.exists(chain_pdbname+".pdb"):
                #     extractPDB(raw_pdbfile, out_filename1+".pdb", chainid)

                # * Compute MSMS of surface w/hydrogens
                vertices1, faces1, normals1, names1, areas1 = computeMSMS(chain_pdbname+".pdb", protonate=True)
                vertex_hbond = computeCharges(chain_pdbname, vertices1, names1)
                
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
                vertex_charges = computeAPBS(regular_mesh.vertices, chain_pdbname+".pdb", chain_pdbname)
                save_ply(ply_file, regular_mesh.vertices,\
                            regular_mesh.faces, normals=vertex_normal, charges=vertex_charges,\
                            normalize_charges=True, hbond=vertex_hbond, hphob=vertex_hphobicity)
                time_end = time.time()
                time_use = time_end - time_start
                # print(f"{pdbid}_{chainid}: using time {time_use}s, finished.")
            # else:
                # print(f"{pdbid}_{chainid}: already exists.")
            # success_protein += 1

        except:
            # set_trace()
            except_pdbs.append(f"{pdbid}_{chainid}")
            continue
    if len(except_pdbs) != 0:
        print(f"except pdbs: {except_pdbs}")
    else:
        print(f"{success_protein} proteins success.")
    

    # write fasta
    # out_fasta = f"{geobind_file.split('.')[0]}.fasta"
    # with open(out_fasta, 'w') as f:
    #     pass 
    # parse_label(geobind_file, dir_opts['chain_pdb_dir'], f"{geobind_file.split('.')[0]}.fasta")