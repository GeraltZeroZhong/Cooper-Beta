import os
import sys
import warnings
import shutil
from Bio import BiopythonWarning
from Bio.PDB import PDBParser, MMCIFParser
from Bio.PDB.DSSP import DSSP
from Bio.PDB.Polypeptide import is_aa
from Bio.SeqUtils import seq1

# Suppress non-critical Biopython warnings (e.g. discontinuous chains)
warnings.simplefilter('ignore', BiopythonWarning)

class ProteinParser:
    """
    Robust loader for PDB/MMCIF files with integrated DSSP execution.
    Handles file format detection and provides clean access to structure/sequence.
    """

    def __init__(self, file_path, model_id=0):
        self.file_path = file_path
        self.model_id = model_id
        self.structure = None
        self.model = None
        self.dssp = None
        self._load_structure()

    def _load_structure(self):
        """Auto-detects format (.pdb/.cif) and loads structure."""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Structure file not found: {self.file_path}")

        # Logic: Detect parser based on extension
        ext = os.path.splitext(self.file_path)[1].lower()
        
        try:
            if ext in ['.cif', '.mmcif']:
                parser = MMCIFParser(QUIET=True)
                self.structure = parser.get_structure('struct', self.file_path)
            else:
                # Default to PDB parser for .pdb, .ent, or no extension
                parser = PDBParser(QUIET=True)
                self.structure = parser.get_structure('struct', self.file_path)
            
            self.model = self.structure[self.model_id]
            
        except Exception as e:
            raise ValueError(f"Failed to parse structure {self.file_path}: {str(e)}")

    def run_dssp(self, dssp_bin=None):
        """
        Executes DSSP.
        Args:
            dssp_bin (str): Path to mkdssp executable. If None, tries to find in PATH.
        Returns:
            dict: The DSSP object (keys: (chain, res_id)).
        """
        if self.dssp is not None:
            return self.dssp

        # 1. Resolve Binary
        if dssp_bin is None:
            # Common names for the binary
            for name in ["mkdssp", "dssp"]:
                if shutil.which(name):
                    dssp_bin = name
                    break
        
        if not dssp_bin or not shutil.which(dssp_bin):
            raise RuntimeError("DSSP binary not found! Please install 'dssp' or 'mkdssp' and add to PATH.")

        # 2. Run DSSP
        try:
            # Note: Biopython's DSSP class handles the temp file creation internally
            self.dssp = DSSP(self.model, self.file_path, dssp=dssp_bin)
            return self.dssp
        except Exception as e:
            raise RuntimeError(f"DSSP execution failed for {self.file_path}: {str(e)}")

    def get_chain(self, chain_id):
        """Safe chain retrieval."""
        if chain_id in self.model:
            return self.model[chain_id]
        return None

    def get_sequence(self, chain_id):
        """
        Extracts the 1-letter amino acid sequence for a chain.
        Useful for checking sequence identity later.
        """
        chain = self.get_chain(chain_id)
        if not chain:
            return ""
        
        seq = []
        for residue in chain:
            # Only count standard amino acids (ignore waters, ligands)
            if is_aa(residue, standard=True):
                seq.append(seq1(residue.get_resname()))
        return "".join(seq)

    def get_ca_coords(self, chain_id):
        """
        Returns list of CA coordinates for a chain.
        Format: [(res_num, [x, y, z]), ...]
        """
        chain = self.get_chain(chain_id)
        if not chain:
            return []
            
        coords = []
        for res in chain:
            if is_aa(res) and 'CA' in res:
                coords.append((res.id[1], res['CA'].get_coord()))
        return coords

# --- Quick Test ---
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python parser.py <pdb_file>")
        sys.exit(1)
        
    pdb_file = sys.argv[1]
    try:
        loader = ProteinParser(pdb_file)
        print(f"Loaded: {os.path.basename(pdb_file)}")
        
        # Try running DSSP
        # Note: You need 'mkdssp' installed for this to work
        # on Ubuntu: sudo apt-get install dssp
        # on Conda: conda install -c salilab dssp
        try:
            dssp_data = loader.run_dssp()
            print(f"DSSP Run Success. Total residues processed: {len(dssp_data)}")
        except RuntimeError as e:
            print(f"DSSP Skipped: {e}")

        # Get Sequence of Chain A
        seq = loader.get_sequence('A')
        print(f"Chain A Sequence ({len(seq)} residues): {seq[:20]}...")
        
    except Exception as e:
        print(f"Error: {e}")