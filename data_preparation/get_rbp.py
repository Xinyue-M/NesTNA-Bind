import sys 
import os 
import Bio.PDB as PDB

sys.path.append("/root/vit")
from default_config.dir_opts import dir_opts
from data_preparation.parse_input import parse_input

from data_preparation.input_output.extractPDB import find_modified_amino_acids
from data_preparation.get_fasta import get_prot_seq

AA_CODES = {
     'ALA':'A', 'CYS':'C', 'ASP':'D', 'GLU':'E',
     'PHE':'F', 'GLY':'G', 'HIS':'H', 'LYS':'K',
     'ILE':'I', 'LEU':'L', 'MET':'M', 'ASN':'N',
     'PRO':'P', 'GLN':'Q', 'ARG':'R', 'SER':'S',
     'THR':'T', 'VAL':'V', 'TYR':'Y', 'TRP':'W', 
     'UNK':'X', 'HID':'H'}

class NBP():

    def __init__(self, pdbid, chainid, rnaids, binding_info, ply_file, pair_pdb, chain_pdb):

        self.pdbid = pdbid
        self.chainid = chainid
        self.rnaids = rnaids 
        self.binding_info = binding_info # golden label
        self.ply_file = ply_file
        self.pair_pdb = pair_pdb
        self.chain_pdb = chain_pdb
        self.parser = PDB.PDBParser(QUIET=True)

        self.seq, self.label = self._get_seq()
    
    def _get_seq_label(self):
        
        resdict = get_prot_seq(self.chain_pdb, self.chainid)
        for (_, info) in enumerate(self.binding_info):
            info = info.split(" ")[1:]
            for site in info:
                if resdict[site[1:]][0] == site[0]:
                    resdict[site[1:]][1] = str(1)
        seq, label = [], []
        for k, v in resdict.items():
            seq.append(v[0])
            label.append(v[1])
        seq = ''.join(seq)
        label = ''.join(label)
        
        return seq, label
    
    def _get_esm_embedding(self):
        
        

        pass 
    
    def _get_surface_res(self):
        pass 



    def get_data(self):
        pass 
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

        nbp = NBP(pdbid, chainid, rnaids, binding_info, ply_file, pair_pdbfile, chain_pdbfile)
