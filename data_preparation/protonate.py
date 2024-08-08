import os 
import sys
import shutil
import Bio.PDB as PDB
from subprocess import Popen, PIPE
from IPython.core.debugger import set_trace

sys.path.append("/root/vit")
from default_config.dir_opts import dir_opts
from default_config.bin_path import bin_path
from data_preparation.split_chain import get_chain

def parse_pdbid_list(geobind_file):

    pdbids = []
    with open(geobind_file, 'r') as f:
        lines = f.read().split('\n')
        for line in lines:
            pdbids.append(line[:4])
    f.close()
    pdbids.remove('')
    return pdbids

def download_pdbfiles(pdbids, dst_dir):

    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    pdbl = PDB.PDBList(server='http://ftp.wwpdb.org')
    for id in pdbids:
        pdb_filename = pdbl.retrieve_pdb_file(pdb_code=id,
                                              pdir=dst_dir,
                                              file_format='pdb')
        shutil.move(pdb_filename, os.path.join(dst_dir, id+'.pdb'))
    pass

def protonate(in_pdb_file, out_pdb_file):
    # protonate (i.e., add hydrogens) a pdb using reduce and save to an output file.
    # in_pdb_file: file to protonate.
    # out_pdb_file: output file where to save the protonated pdb file. 
    
    # Remove protons first, in case the structure is already protonated
    args = ["reduce", "-Trim", in_pdb_file]
    p2 = Popen(args, stdout=PIPE, stderr=PIPE)
    stdout, stderr = p2.communicate()
    outfile = open(out_pdb_file, "w")
    outfile.write(stdout.decode('utf-8').rstrip())
    outfile.close()
    # Now add them again.
    args = ["reduce", "-HIS", out_pdb_file]
    p2 = Popen(args, stdout=PIPE, stderr=PIPE)
    stdout, stderr = p2.communicate()
    outfile = open(out_pdb_file, "w")
    outfile.write(stdout.decode('utf-8'))
    outfile.close()


