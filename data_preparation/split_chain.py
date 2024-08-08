import os 
import sys 
import shutil
from Bio import PDB

class ChainSplitter:
    def __init__(self, out_dir=None):
        """ Create parsing and writing objects, specify output directory. """
        self.parser = PDB.PDBParser()
        self.writer = PDB.PDBIO()
        if out_dir is None:
            out_dir = os.path.join(os.getcwd(), "chain_PDBs")
        self.out_dir = out_dir

    def make_pdb(self, pdb_id, pdb_fn, chain_letters, overwrite=False, struct=None):
        """ Create a new PDB file containing only the specified chains.

        Returns the path to the created file.

        :param pdb_path: full path to the crystal structure
        :param chain_letters: iterable of chain characters (case insensitive)
        :param overwrite: write over the output file if it exists
        """
        chain_letters = [chain for chain in chain_letters]

        # Input/output files
        # (pdb_dir, pdb_fn) = os.path.split(pdb_path)
        # pdb_id = pdb_fn[:4]
        out_name = "%s_%s.pdb" % (pdb_id, "".join(chain_letters))
        out_path = os.path.join(self.out_dir, out_name)
        # print "OUT PATH:",out_path
        plural = "s" if (len(chain_letters) > 1) else ""  # for printing

        # Skip PDB generation if the file already exists
        if (not overwrite) and (os.path.isfile(out_path)):
            print("Chain%s %s of '%s' already extracted to '%s'." %
                    (plural, ", ".join(chain_letters), pdb_id, out_name))
            return out_path

        print("Extracting chain%s %s from %s..." % (plural,
                ", ".join(chain_letters), pdb_fn))

        # Get structure, write new file with only given chains
        if struct is None:
            struct = self.parser.get_structure(pdb_id, pdb_fn)
        self.writer.set_structure(struct)
        self.writer.save(out_path, select=SelectChains(chain_letters))

        return out_path

class SelectChains(PDB.Select):
    """ Only accept the specified chains when saving. """
    def __init__(self, chain_letters):
        self.chain_letters = chain_letters

    def accept_chain(self, chain):
        return (chain.get_id() in self.chain_letters)

def removeHETATM(pdb_fn, out_fn):
    with open(pdb_fn, 'r') as f:
        lines = f.readlines()
        with open(out_fn, 'w') as g:
            for line in lines:
                if not line.startswith("HETATM"):
                    g.write(line)

def get_chain(pdbid, proid, rnaid, src_dir, dst_dir):
    """
    Get pdbid, chainid from the raw interaction txt of GEOBIND.
    """

    splitter = ChainSplitter(out_dir=dst_dir)    
    # try:
    rnaids = rnaid.split("_")
    # except:
        # ranids = rnaid
    print(rnaids)
    raw_pdb_file = os.path.join(src_dir, f"{pdbid}.pdb")
    pair_pdb = splitter.make_pdb(pdbid, raw_pdb_file, chain_letters=[proid] + rnaids, overwrite=False)
    chain_file = os.path.join(dst_dir, f"{pdbid}_{proid}_{''.join(rnaid)}.pdb")
    removeHETATM(pair_pdb, chain_file)
    os.remove(pair_pdb)

def get_pair_complex(info_dict, pdb_dir, out_dir):
    """
    From the raw pdb files, get a complex of protein and RNA.
    """
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    splitter = ChainSplitter(out_dir=out_dir)    
    for k, _ in info_dict.items():
        [pdbid, proid, rnaid] = k.split('_')
        raw_pdb_file = os.path.join(pdb_dir, pdbid + ".pdb")
        # out_fn =  os.path.join(pdb_dir, pdbid + "_nohetatm.pdb")
        # removeHETATM(raw_pdb_file, out_fn)
        splitter.make_pdb(pdbid, raw_pdb_file, chain_letters=[proid, rnaid], overwrite=False)
