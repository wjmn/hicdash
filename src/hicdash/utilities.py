"""Very basic package-wide utility functions for e.g. simple unit conversions.
"""

import matplotlib.pyplot as plt
import pyensembl
from hicstraw import HiCFile

def chr_unprefix(chr_string: str) -> str:
    """Remove the "chr" prefix from a chromosome string.

    If no "chr" prefix is present, returns the string unchanged.
    """
    if chr_string.startswith("chr"):
        return chr_string[3:]
    return chr_string


def chr_prefix(chr_string: int | str) -> str:
    """Add the "chr" prefix to a chromosome if not already included.

    If the "chr" prefix is already present, returns the string unchanged.
    """
    if str(chr_string).startswith("chr"):
        return str(chr_string)
    return f"chr{str(chr_string)}"


def to_mega(number: int, dp=1) -> float:
    """Divide by a million to a given number of decimal places.
    
    Useful e.g. to convert base pairs to megabases with dp (1dp by default.)
    """
    return round(number / 1e6, dp)


def resolution_to_int(suffixed_str: str | int) -> int:
    """Convert a string with a "Mb" or "kb" suffix to an int.
    """
    if isinstance(suffixed_str, int):
        return suffixed_str
    if suffixed_str.endswith("Mb"):
        return int(float(suffixed_str[:-2]) * 1e6)
    elif suffixed_str.endswith("kb"):
        return int(float(suffixed_str[:-2]) * 1e3)
    else:
        return int(float(suffixed_str))

def is_protein_coding(gene: pyensembl.Gene) -> bool:
    return gene.biotype == "protein_coding"

def get_gene_name_or_id(gene: pyensembl.Gene) -> str:
    if gene.gene_name == "":
        return gene.gene_id
    return gene.gene_name

def is_ig_gene(gene: pyensembl.Gene) -> bool:
    return (gene.biotype == "IG_V_gene" or gene.biotype == "IG_D_gene" or gene.biotype == "IG_J_gene" or gene.biotype == "IG_C_gene")


def int_to_resolution(resolution: int) -> str:
    """Convert an int resolution into a suffixed with either "kb" or "Mb".

    Rounds to the nearest whole thousand (kb) or million (Mb). 

    """
    if resolution >= 1000000:
        if resolution % 1000000 == 0:
            return f"{resolution // 1000000}Mb"
        else:
            return f"{resolution / 1000000:.1f}Mb"
    elif resolution >= 1000:
        return f"{resolution // 1000}kb"
    else:
        return f"{resolution}b"

def read_hic(hic_file: str) -> HiCFile:
    """Read Hi-C file and return a HiCFile object.
    
    This is just a convenience wrapper for now, to be revised. 
    """
    return HiCFile(hic_file)

def blank_axis(ax: plt.Axes):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[['top', 'right', 'left', 'bottom']].set_visible(False)