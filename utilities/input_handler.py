from Bio.PDB import *

from Bio import SeqIO
import re


ATOM_TO_HYDROPHOBICITY = {
    'G': 'H',
    'A': 'H',
    'V': 'H',
    'L': 'H',
    'I': 'H',
    'P': 'H',
    'F': 'H',
    'M': 'H',
    'W': 'H',
    'S': 'P',
    'T': 'P',
    'C': 'P',
    'N': 'P',
    'Q': 'P',
    'Y': 'P',
}

d3to1 = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
         'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
         'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
         'ALA': 'A', 'VAL': 'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}


def extract_hydropolar_sequence_from_pdb(file_name):
    with open(file_name) as f:
        for matched_line in re.findall('^SEQRES.*$', f.read(), re.MULTILINE):
            tokens = matched_line.split()
            print(tokens)

def extract_hydropolar_sequence_from_fasta(file_name):
    amino_acid_sequence = ""
    index = 0
    for seq_record in SeqIO.parse(file_name, "fasta"):
        if index > 0:
            print(
                "Warning: The model works only with single-chain proteins. The input will be truncated to use the first chain. ")
            break
        sequence = seq_record.seq
        amino_acid_sequence = ''.join([ATOM_TO_HYDROPHOBICITY[t] for t in sequence])
        index += 1

    print(amino_acid_sequence)
    return amino_acid_sequence


extract_hydropolar_sequence_from_pdb(
    "/Users/ancaioanamuscalagiu/Documents/licenta/ProteinFolding/known_proteins/pdb/110d.pdb")
