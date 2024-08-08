import os
import sys
import Bio.PDB as PDB

# AA_CODES = {
#      'ALA':'A', 'CYS':'C', 'ASP':'D', 'GLU':'E',
#      'PHE':'F', 'GLY':'G', 'HIS':'H', 'LYS':'K',
#      'ILE':'I', 'LEU':'L', 'MET':'M', 'ASN':'N',
#      'PRO':'P', 'GLN':'Q', 'ARG':'R', 'SER':'S',
#      'THR':'T', 'VAL':'V', 'TYR':'Y', 'TRP':'W', 
#      'UNK':'X', 'HID':'H'}
AA_CODES = {
     'ALA':'A', 'CYS':'C', 'ASP':'D', 'GLU':'E',
     'PHE':'F', 'GLY':'G', 'HIS':'H', 'LYS':'K',
     'ILE':'I', 'LEU':'L', 'MET':'M', 'ASN':'N',
     'PRO':'P', 'GLN':'Q', 'ARG':'R', 'SER':'S',
     'THR':'T', 'VAL':'V', 'TYR':'Y', 'TRP':'W', 
     'HID':'H'}
def get_prot_seq(chain_pdbfile, proid):
    '''
    Format of pdb_id: xxxx_(chain)
    '''
    resdict = {}
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure(file=chain_pdbfile, id=proid)
    model = structure.get_models()
    model = list(model)
    model = model[0]

    chains = model.get_chains()
    for _, chain in enumerate(chains):
        if chain.id == proid:
            residues = list(chain.get_residues())
            for (idx, res) in enumerate(residues):
                if res.get_resname() in AA_CODES.keys():
                    resname = AA_CODES[res.get_resname()] # 1-letter
                else:
                    continue
                    # resname = 'X'
                resnum = res.get_full_id()[3][1]
                resdict[str(resnum)] = [resname, str(0)]
    return resdict

def parse_label(geobind_file, srcdir, out_fasta):

    labels = []
    out_f = open(out_fasta, 'w')
    # for (_, info) in enumerate(input_info):
    #     pdbid, chainid, rnaids, binding_info = info

    with open(geobind_file, 'r') as f:
        lines = f.read().split("\n")
        for line in lines:
            # * dmasif file
            # pdbid, chainid, rnaids = line.split('_')
            # * geobind file
            pdbid, chainid, rnaids = line.split("\t")[0].split(':')
            pdb_chain = f"{pdbid}_{chainid}"
            print(pdbid, chainid, rnaids)
            # try:
            if os.path.exists(os.path.join(srcdir, f"{pdb_chain}.pdb")):
                resdict = get_prot_seq(chain_pdbfile=os.path.join(srcdir, f"{pdb_chain}.pdb"), proid=chainid)
                binding_info = line.split("\t")[1].split(':')
                for info in binding_info:
                    info = info.split(" ")[1:]
                    for site in info:
                        # assert resdict[site[1:]][0] == site[0]
                        try:
                            if resdict[site[1:]][0] == site[0]:
                                resdict[site[1:]][1] = str(1)
                        except: 
                            print(f"ERROR:{pdb_chain}")
                # print(resdict)
                seq, label = [], []
                for k, v in resdict.items():
                    seq.append(v[0])
                    label.append(v[1])
                seq = ''.join(seq)
                label = ''.join(label)
                out_f.write(f">{pdb_chain}\n")
                out_f.write(f"{seq}\n")
                # out_f.write(f"{label}\n")

            # break

def write_fasta(pdbid, chainid, rnaids, binding_info, chain_pdbfile, out_f):
    
    resdict = get_prot_seq(chain_pdbfile, chainid)
    for info in binding_info:
        info = info.split(" ")[1:]
        # print(info)
        for site in info:
            # assert resdict[site[1:]][0] == site[0]
            try:
                if resdict[site[1:]][0] == site[0]:
                    resdict[site[1:]][1] = str(1)
            except: 
                print(f"ERROR:{pdbid}")  
                print(site)

    seq, label = [], []
    for k, v in resdict.items():
        seq.append(v[0])
        label.append(v[1])
    seq = ''.join(seq)
    label = ''.join(label)
    pdb_chain = f"{pdbid}_{chainid}"
    out_f.write(f">{pdb_chain}\n")
    out_f.write(f"{seq}\n")
    out_f.write(f"{label}\n")

if __name__ == "__main__":

    print("start")
    geobind_file = "DNA-179_Test.txt"
    # geobind_file = "DNA-719_Train.txt"
    # geobind_file = "RNA-157_Test.txt"
    # geobind_file = "RNA-663_Train.txt"
    # geobind_file = "RNA_test_ours.txt"
    # geobind_file = "testing_ppi.txt"
    
    # srcdir = "data/chain_pdb/rbp_our"
    srcdir = os.path.join("data/chain_pdb", f"{geobind_file.split('.')[0]}")
    # write_fasta()
    # pdbid = "1A6B_B"
    # get_prot_seq(pdbid, srcdir)
    parse_label(geobind_file, srcdir, f"{geobind_file.split('.')[0]}.fasta")