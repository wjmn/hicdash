"""Useful package-wide basic types.

These types help provide the basis for useful type signatures for 
functions and classes in the package, as well as type hints 
during development. 

"""
from __future__ import annotations
from typing import Any
from enum import Enum
from dataclasses import dataclass, replace
from typing_extensions import Self
from hicstraw import HiCFile # type: ignore
from numpy.typing import NDArray
import pandas as pd
import numpy as np
from io import StringIO
import pyensembl # type: ignore

from hicdash.constants import CHROM_INDICES, CHROM_SIZES, BREAKFINDER_COLUMNS, BEDPE_COLUMNS, GENE_ANNOTATIONS, CHROMS
from hicdash.utilities import chr_prefix, chr_unprefix, is_protein_coding, resolution_to_int, read_hic



class Strand(Enum):
    """A genomic strand; can be either POSitive (forward) or NEGative (minus). 
    """
    POS = "+"
    NEG = "-"

    @staticmethod
    def from_string(string: str) -> Strand:
        """Convert '+' or '-' to Strand, raise ValueError if neither."""
        match string:
            case "+": 
                return Strand.POS
            case "-":
                return Strand.NEG 
            case _:
                raise ValueError(f"Invalid strand: {string} (expected + or -).")
                
    def __repr__(self) -> str:
        """Represented as simply '+' or '-'."""
        match self:
            case Strand.POS:
                return "+"
            case Strand.NEG:
                return "-"
            case _:
                raise ValueError(f"Unrecognized strand: {self}")

    def __str__(self) -> str:
        """Converted to string as simply '+' or '-'."""
        return self.__repr__()

    def opposite(self) -> Strand:
        """Return the opposite strand."""
        match self:
            case Strand.POS:
                return Strand.NEG
            case Strand.NEG:
                return Strand.POS
            case _: # unreachable
                raise ValueError(f"Unrecognized strand: {self}")


@dataclass
class GenomicRegion:
    """An interval on the genome, consisting of a chromosome, start point and end point. 

    Optionally specify if this region is tagged as reverse (default False), and if the region is aligned to a certain resolution.

    Properties:
    - chrom: str: Chromosome name (prefixed)
    - start: int: Start point of the region
    - end: int: End point of the region
    - reverse: bool: Whether the region is reversed (default False)
    - bin_align_res: int | None: If not None, the region is aligned to this resolution (i.e. start and end are multiples of this resolution)
    """
    
    chrom: str 
    start: int 
    end: int 
    reverse: bool = False
    bin_align_res: int | None = None
    
    def __init__(self, chrom: int | str, start: int, end: int, reverse=False, bin_align_res=None):
        """Initialize a new genomic region.

        If bin_align_res is not None, will align start to the start of the bin containing start, and end to the end of the bin containing end (inclusive). 

        If start and end are already multiples of bin_align_res, this will not change the region.

        Params:
        - chrom: int | str: Chromosome name (prefixed)
        - start: int: Start point of the region
        - end: int: End point of the region
        - reverse: bool: Whether the region is reversed (default False)
        - bin_align_res: int | None: If not None, the region is aligned to this resolution (i.e. start and end are multiples of this resolution)

        """

        self.chrom: str = chr_prefix(str(chrom))
        self.start: int = int(start)
        self.end: int = int(end)

        if bin_align_res is not None:
            # If already multiples of bin_align_res, this calculation won't do anything
            self.start = self.start // bin_align_res * bin_align_res
            self.end  = (self.end - 1) // bin_align_res * bin_align_res + bin_align_res
            self.bin_align_res = bin_align_res

        self.reverse: bool = reverse

    def __str__(self) -> str:
        """Formatted as chrom:start-end (reversed if reverse is True)."""
        return f"{self.chrom}:{self.start}-{self.end}{' (reversed)' if self.reverse else ''}"

    def __lt__(self, other: Any) -> bool:
        """Sort purely based on chrom index, start point, end point (in that order)."""
        if not isinstance(other, GenomicRegion):
            return NotImplemented
        index_self = CHROM_INDICES[self.chrom]
        index_other = CHROM_INDICES[other.chrom]
        if index_self == index_other:
            if self.start == other.start:
                return self.end < other.end
            else:
                return self.start < other.start
        else:
            return index_self < index_other

    def __le__(self, other: Any) -> bool:
        """Sort purely based on chrom index, start point, end point (in that order)."""
        if not isinstance(other, GenomicRegion):
            return NotImplemented
        index_self = CHROM_INDICES[self.chrom]
        index_other = CHROM_INDICES[other.chrom]
        if index_self == index_other:
            if self.start == other.start:
                return self.end <= other.end
            else:
                return self.start <= other.start
        else:
            return index_self <= index_other
            

    def get_bin_aligned(self, resolution: int | None) -> GenomicRegion:
        """Align the region endpoints to multiples of the provided resolution and return a new region. Does NOT modify the original region.
        Guaranteed that new start <= old start and new end >= original end (i.e. will only extend the region)
        Extends the region to the end of the bin that contains the endpoint
        If resolution is None, returns the region unchanged.
        """
        if resolution is None:
            return self
        start = self.start // resolution * resolution
        end = (self.end-1) // resolution * resolution + resolution
        return replace(self, start=start, end=end, bin_align_res=resolution)

    def get_contained_genes(self, strand: Strand | None=None) -> list[pyensembl.Gene]:
        """Return all genes (as pyensembl gene objects) that overlap with this region. Optionally filter by strand."""
        strand_string = str(strand) if strand is not None else None # pyensembl expects + or - 
        return GENE_ANNOTATIONS.genes_at_locus(chr_unprefix(self.chrom), self.start, end=self.end, strand=strand_string)

    def get_reverse_true(self) -> GenomicRegion:
        """Return a new genomic region with reverse set to True (regardless of what it was before)."""
        return replace(self, reverse=True)

    def get_opposite(self) -> GenomicRegion:
        """Return a new genomic region with reverse set to the opposite of what it was before."""
        return replace(self, reverse=not self.reverse)

    def get_size(self) -> int:
        """Return the size of the region."""
        return self.end - self.start

    def get_center(self) -> int:
        """Return the center of the region (truncated down to integer)"""
        return (self.end + self.start) // 2

    def get_unpacked(self) -> tuple[str, int, int]:
        """Unpack into (chrom, start, end).  """
        return (self.chrom, self.start, self.end)

    def get_unpacked_full(self) -> tuple[str, int, int, bool, int | None]:
        """Unpack into (chrom, start, end, reverse, bin_align_res)"""
        return (self.chrom, self.start, self.end, self.reverse, self.bin_align_res)

    def overlaps(self, other: GenomicRegion) -> bool:
        """Return True if this region overlaps with the other region."""
        if ( self.chrom == other.chrom and 
            (
                self.start <= other.start <= self.end or 
                other.start <= self.start <= other.end or 
                self.start <= other.end <= self.end or 
                other.start <= self.end <= other.end
            )):
            return True   
        return False

    @staticmethod
    def from_chrom(chrom: str) -> GenomicRegion:
        prefixed = chr_prefix(chrom)
        return GenomicRegion(prefixed, 0, CHROM_SIZES[prefixed])



@dataclass
class GenomicPosition:
    """A single coordinate on the genome (chromosome and position)."""
    
    chrom: str
    pos: int
    
    def __init__(self, chrom: int | str, pos: int):
        """Initialize a new genomic position.

        Ensures at initialization that 
        1. Chrom is always in prefixed form.
        2. Position is always an int

        """
        self.chrom: str = chr_prefix(str(chrom))
        self.pos: int = int(pos)
        
    def __str__(self) -> str:
        """Formatted as chrom:pos."""
        return f"[{self.chrom}:{self.pos}]"

    def __lt__(self, other: Any) -> bool:
        """Sort purely based on chrom index, then position."""
        if not isinstance(other, GenomicPosition):
            return NotImplemented
        self_index = CHROM_INDICES[self.chrom]
        other_index = CHROM_INDICES[other.chrom]
        if self_index == other_index:
            return self.pos < other.pos
        return self_index < other_index

    def __le__(self, other: Any) -> bool:
        """Sort purely based on chrom index, then position."""
        if not isinstance(other, GenomicPosition):
            return NotImplemented
        self_index = CHROM_INDICES[self.chrom]
        other_index = CHROM_INDICES[other.chrom]
        if self_index == other_index:
            return self.pos <= other.pos
        return self_index <= other_index
            
    def get_intersecting_genes(self, strand: Strand | None=None) -> list[pyensembl.Gene]:
        """Return all genes (as pyensembl gene objects) that intersect with this position. Optionally filter by strand."""
        strand_string = str(strand) if strand is not None else None
        return GENE_ANNOTATIONS.genes_at_locus(chr_unprefix(self.chrom), self.pos, strand=strand_string)
        
    def get_region_with_symmetrical_radius(self, radius: int, bin_align_res: int | None = None) -> GenomicRegion:
        """Return a genomic region centered around this position with a given radius on either side.
        
        Optionally choose to bin-align this region to a resolution. The radius must be a multiple of the bin align resolution.

        Will align by aligning the startpoint first, then adding twice the radius. 
        """
        if bin_align_res is None:
            return GenomicRegion(self.chrom, self.pos - radius, self.pos + radius)
        else:
            # Bin align one endpoint first, then add the radius * 2 to ensure width is exactly radius*2
            # (assuming radius % bin_align_res == 0)
            # This is a centered region, so likely the actual endpoints don't matter as much as what's inside it
            assert radius % bin_align_res == 0
            start = (self.pos - radius) // bin_align_res * bin_align_res
            end = start + radius * 2
            return GenomicRegion(self.chrom, start, end, bin_align_res=bin_align_res)


    def get_unpacked_pair(self) -> tuple[str, int]:
        """Unpacks into (chrom, pos)"""
        return (self.chrom, self.pos)
        
    def is_inside(self, gregion: GenomicRegion) -> bool:
        """Returns True if this position is inside the given genomic region (inclusive endpoints)"""
        if self.chrom == gregion.chrom and gregion.start <= self.pos <= gregion.end:
            return True
        return False
        


@dataclass
class Breakend(GenomicPosition): 
    """A stranded single breakend on the genome (chrom, pos, strand).

    Inherits from GenomicPosition (with additional strand tag). 

    """
    
    chrom: str
    pos: int
    strand: Strand
    
    def __init__(self, chrom: int| str, pos: int, strand: Strand | str):
        """Iniiialize a new breakend.

        Strand can be provided as either Strand enum or as a string ('+' or '-').

        """
        self.chrom = chr_prefix(str(chrom))
        self.pos = int(pos)
        self.strand = strand if isinstance(strand, Strand) else Strand.from_string(strand)

    def __str__(self) -> str:
        """Formatted as [chrom:pos:strand]."""
        return f"[{self.chrom}:{self.pos}:{self.strand}]"

    def get_bare_string(self) -> str:
        """Return a string representation without the brackets and without the strand (i.e. chrom:pos)."""
        return f"{self.chrom}:{self.pos}"

    def get_unpacked(self) -> tuple[str, int, Strand]:
        """Unpacks into (chrom, pos, strand)"""
        return (self.chrom, self.pos, self.strand)
        
    def get_region_with_stranded_radius(self, radius: int, bin_align_res: int | None = None) -> GenomicRegion:
        """Return a genomic region of a given size (radius) with the breakend at one end depending on strandness.
        
        If the breakend strand is positive, the breakend position is at the end of the region.
        If thee breakend strand is negative, the breakend position is at the start of the region.

        If bin_align_res is specified, then the region is aligned to the resolution (i.e. start and end are multiples of the resolution).
        The breakend position is aligned first, then the position is extended by the radius in the appropriate direction.
        Radius must be a multiple of bin_align_res. 
        """
        match self.strand:
            case Strand.POS:
                if bin_align_res is None:
                    return GenomicRegion(self.chrom, self.pos - radius, self.pos)
                else:
                    # The endpoint is the crucial point here-  ensure that enpdoint is captured in bin, then subtract radius
                    assert radius % bin_align_res == 0
                    end = (self.pos - 1) // bin_align_res * bin_align_res + bin_align_res
                    start = end - radius
                    return GenomicRegion(self.chrom, start, end, bin_align_res=bin_align_res)
            case Strand.NEG:
                if bin_align_res is None:
                    return GenomicRegion(self.chrom, self.pos, self.pos + radius)
                else:
                    # The startpoint is the crucial point here - ensure startpoint is captured
                    assert radius % bin_align_res == 0
                    start = self.pos // bin_align_res * bin_align_res
                    end = start + radius
                    return GenomicRegion(self.chrom, start, end, bin_align_res=bin_align_res)
                    
    def get_region_with_symmetrical_radius(self, radius: int, bin_align_res: int | None = None) -> GenomicRegion:
        """Return a genomic region of a given size (2 * radius) centered around the breakend position.

        Overriden from GenomicRegion to ensure consistent with stranded radius bounds.
        Essentially find stranded radius bounds, then depending on strand, extend the radius
        """
        match self.strand:
            case Strand.POS:
                gregion = self.get_region_with_stranded_radius(radius, bin_align_res)
                gregion = replace(gregion, end=gregion.end + radius)
                return gregion
            case Strand.NEG:
                gregion = self.get_region_with_stranded_radius(radius, bin_align_res)
                gregion = replace(gregion, start=gregion.start - radius)
                return gregion


@dataclass
class PlotRegion:
    """A region representing a bin range on a plot (in axis coordinates) as well as the genomic region is represents (or None).

    Properties:
    - bin_range: tuple[int, int]: The bin range on the plot (endpoints are inclusive)
    - genomic_region: GenomicRegion | None: The genomic region this plot region represents (if any). None represents a plot gap.
    """
    bin_range: tuple[int, int]
    genomic_region: GenomicRegion | None


@dataclass
class AssembledHic: 
    """Returns assembled Hi-C data as well as a list of the oriented Hi-C regions it represents (in successive order).
    
    The Hi-C matrix data is stored as a single array in the data attribute, and the regions are stored in plot_regions (which show correspondence between bins and genomic regions).

    Properties:
    - data: NDArray: The Hi-C matrix data
    - plot_regions: list[PlotRegion]: The regions represented in the Hi-C matrix data (in order, oriented)

    """
    data: NDArray
    plot_regions: list[PlotRegion]


@dataclass
class PairedRegion:
    """A generic paired region, similar to what might be read from a bedpe file.

    Properties:
    - regionA: GenomicRegion: The first region
    - regionB: GenomicRegion: The second region
    """
    regionA: GenomicRegion
    regionB: GenomicRegion

    def __init__(self, chromA: int | str, startA: int, endA: int, chromB: int | str, startB: int, endB: int):
        """Initialize a new paired region simply i.e. PairedRegion(chromA, startA, endA, chromB, startB, endB).
        
        Ensures at initialization that regionA <= regionB.
        """
        regionA = GenomicRegion(chromA, startA, endA)
        regionB = GenomicRegion(chromB, startB, endB)
        if regionB < regionA:
            regionA, regionB = regionB, regionA
        self.regionA = regionA
        self.regionB = regionB
        
    def __lt__(self, other: Any):
        """Sort by breakendA, then breakendB."""
        if not isinstance(other, PairedRegion):
            return NotImplemented
        if self.regionA == other.regionA:
            return self.regionB < other.regionB
        else:
            return self.regionA < other.regionA

    def __le__(self, other: Any):
        """Sort by breakendA, then breakendB."""
        if not isinstance(other, PairedRegion):
            return NotImplemented
        if self.regionA == other.regionA:
            return self.regionB <= other.regionB
        else:
            return self.regionA <= other.regionA
        

    @classmethod
    def from_bedpe_row(cls, row: Any):
        """Create a PairedRegion object from a row of a bedpe file.
        
        Expects row to have keys "chr1", "start1", "end1", "chr2", "start2", "end2".
        """
        chromA = chr_prefix(row["chr1"])
        startA = row["start1"]
        endA = row["end1"]
        chromB = chr_prefix(row["chr2"])
        startB = row["start2"]
        endB = row["end2"]
        return cls(chromA, startA, endA, chromB, startB, endB)
    
    @classmethod
    def list_from_bedpe_file(cls, filepath: str, skiprows=0) -> list:
        """Returns a sorted list of PairedRegion objects from a bedpe file.

        Expects file to have at least six columns: chr1, start1, end1, chr2, start2, end2.
        """
        df = pd.read_csv(filepath, sep="\s+", names=BEDPE_COLUMNS, skiprows=skiprows, usecols=[0, 1, 2, 3, 4, 5])
        pairs = [cls.from_bedpe_row(row) for _, row in df.iterrows()]
        pairs.sort()
        return pairs


@dataclass
class Breakpoint:
    """A stranded breakpoint consisting of the joining of two breakends.

    Hi-C breakfinder calls SVs as ranges - these are stored as break_regionA and break_regionB (otherwise they are None).

    Ensures breakendA < breakendB at initialization time.

    Properties:
    - breakendA: Breakend: The first breakend
    - breakendB: Breakend: The second breakend
    - resolution: int | None: The resolution of the breakpoint (if known)
    - neg_log_pval: float | None: The negative log p-value of the breakpoint (if known)
    - break_regionA: GenomicRegion | None: The region of the first breakend (if hic_breakfinder call)
    - break_regionB: GenomicRegion | None: The region of the second breakend (if hic_breakfinder call)


    """

    breakendA: Breakend
    breakendB: Breakend
    resolution: int | None=None
    neg_log_pval: float | None=None
    break_regionA: GenomicRegion | None=None
    break_regionB: GenomicRegion | None=None

    def __init__(self, chromA: int | str, posA: int, strandA: Strand | str, chromB: int | str, posB: int, strandB: Strand | str, resolution: int | None = None, neg_log_pval: float | None = None, 
                break_regionA: GenomicRegion | None=None, break_regionB: GenomicRegion | None = None):
        """Initialize a breakpoint simply i.e. Breakpoint(chromA, posA, strandA, chromB, posB, strandB)
        Ensures breakendA < breakendB at initialization time.
        """
        breakendA = Breakend(chromA, posA, strandA)
        breakendB = Breakend(chromB, posB, strandB)

        # Ensure breakendA < breakendB
        if breakendB < breakendA:
            breakendB, breakendA = breakendA, breakendB
            break_regionA, break_regionB = break_regionB, break_regionA
            
        self.breakendA = breakendA
        self.breakendB = breakendB
        self.resolution = resolution
        self.neg_log_pval = neg_log_pval
        self.break_regionA = break_regionA
        self.break_regionB = break_regionB

    def __lt__(self, other: Any):
        """Sort by breakendA, then breakendB."""
        if not isinstance(other, Breakpoint):
            return NotImplemented
        if self.breakendA == other.breakendA:
            return self.breakendB < other.breakendB
        else:
            return self.breakendA < other.breakendA

    def __le__(self, other: Any):
        """Sort by breakendA, then breakendB."""
        if not isinstance(other, Breakpoint):
            return NotImplemented
        if self.breakendA == other.breakendA:
            return self.breakendB <= other.breakendB
        else:
            return self.breakendA <= other.breakendA
        

    def __str__(self) -> str:
        """Formatted as a pictorial representation of the breakpoint with direction of strand marked by < and >."""
        strA = self.breakendA.get_bare_string()
        strB = self.breakendB.get_bare_string()
        
        # Provide an intuitive depiction of strandness
        match self.breakendA.strand:
            case Strand.POS:
                strA = f"[{strA}>"
            case Strand.NEG:
                strA = f"<{strA}]"
        match self.breakendB.strand:
            case Strand.POS:
                strB = f"<{strB}]"
            case Strand.NEG:
                strB = f"[{strB}>"
                
        return f"{strA}::{strB}"

    def get_reconstructed_regions_with_radius(self, radius: int, bin_align_res: int | None=None) -> tuple[GenomicRegion, GenomicRegion]:
        """Converts breakpoint to oriented left and right regions of size radius around the breakpoint, where the left region joins with the right region"""
        region_left = self.breakendA.get_region_with_stranded_radius(radius, bin_align_res)
        region_right = self.breakendB.get_region_with_stranded_radius(radius, bin_align_res)
        # The left region has its breakpoint depicted on the right - so neg strand needs to reverse
        if self.breakendA.strand == Strand.NEG:
            region_left = region_left.get_reverse_true()
        # The right region has its breakpoint depicted on the left - so pos strand needs to reverse
        if self.breakendB.strand == Strand.POS:
            region_right = region_right.get_reverse_true()
        return (region_left, region_right)

    def get_assembly(self, radius: int, bin_align_res: int | None=None) -> list[GenomicRegion]:
        """Get reconstructed left and right regions as a list/assembly."""
        region_left, region_right = self.get_reconstructed_regions_with_radius(radius, bin_align_res=bin_align_res)
        return [region_left, region_right]

    def get_centered_regions_with_radius(self, radius: int, bin_align_res: int | None=None) -> tuple[GenomicRegion, GenomicRegion]:
        """Converts breakpoint to two regions, each centered around each breakend (useful for matrix visualization)"""
        regionA = self.breakendA.get_region_with_symmetrical_radius(radius, bin_align_res)
        regionB = self.breakendB.get_region_with_symmetrical_radius(radius, bin_align_res)
        return (regionA, regionB)

    def get_strandness_type(self) -> str:
        """Predict SV type based on strandness ('translocation', 'deletion', 'duplication' or 'inversion or other'). 
        Note: not always accurate (especially for large SVs).
        All inter-chromosomal SVs are designated "translocation". 
        +/- is designated deletion. 
        -/+ is designated duplication. 
        Any other intra-chromosomal SVs are designated inversion or other. 

        See:
        Song F, Xu J, Dixon J, Yue F. Analysis of Hi-C Data for Discovery of Structural Variations in Cancer. Methods Mol Biol. 2022;2301:143-161.
        """
        if self.breakendA.chrom != self.breakendB.chrom:
            return "translocation"
        match (self.breakendA.strand, self.breakendB.strand):
            case (Strand.POS, Strand.NEG):
                return "deletion"
            case (Strand.NEG, Strand.POS):
                return "duplication"
            case _:
                return "inversion or other"

    def is_inter(self) -> bool:
        """Return True if the breakpoint is inter-chromosomal."""
        return self.breakendA.chrom != self.breakendB.chrom

    def get_deleted_region(self) -> GenomicRegion | None:
        """Return the region that is between the two breakends if the breakpoint strandness matches a deletion, otherwise None."""
        if self.breakendA.chrom != self.breakendB.chrom:
            return None
        else:
            if self.get_strandness_type() != "deletion": 
                return None
            return GenomicRegion(self.breakendA.chrom, self.breakendA.pos, self.breakendB.pos)

    def get_possible_gene_fusions(self) -> list[tuple[pyensembl.Gene, pyensembl.Gene]]:
        """Get a list of possible gene fusions, returning a list of pairs of pyensembl gene objects.
        Here, a possible gene fusion satisfies the following criteria:
        1. Each breakend occurs directly within a protein-coding gene
        2. The orientation of the strands is such that there is continuity
           between the promoter of one gene to the end of the other gene
           (i.e. once the strands are oriented side-by-side, the genes must match orientation")
        """
        genesA = [g for g in self.breakendA.get_intersecting_genes() if is_protein_coding(g) ]
        genesB = [g for g in self.breakendB.get_intersecting_genes() if is_protein_coding(g) ]
        
        genesA_forward = [ gene for gene in genesA if gene.strand == "+"]
        genesA_backward = [ gene for gene in genesA if gene.strand == "-"]
        genesB_forward = [ gene for gene in genesB if gene.strand == "+"]
        genesB_backward = [ gene for gene in genesB if gene.strand == "-"]

        # Swap if the orientation of the strands reverses when put side by side
        if self.breakendA.strand == Strand.NEG:
            genesA_forward, genesA_backward = genesA_backward, genesA_forward
        if self.breakendB.strand == Strand.POS:
            genesB_forward, genesB_backward = genesB_backward, genesB_forward

        possible_fusions_forward = [  (geneA, geneB) for geneA in genesA_forward for geneB in genesB_forward ]
        possible_fusions_backward = [  (geneB, geneA) for geneA in genesA_backward for geneB in genesB_backward ]

        return possible_fusions_forward + possible_fusions_backward

    def get_unpacked(self) -> tuple[str, int, Strand, str, int, Strand]:
        """Unpack the breakpoint into its components (chromA, posA, strandA, chromB, posB, strandB)."""
        return (self.breakendA.chrom, self.breakendA.pos, self.breakendA.strand, self.breakendB.chrom, self.breakendB.pos, self.breakendB.strand)

    def get_nearby_key_genes(self, radius: int, key_genes: list[str]) -> tuple[list[str], list[str]]:
        """Find nearby key genes (using in direction of stranded regions), returning a list of gene NAMES."""
        key_genes_set = set(key_genes)
        region_left, region_right = self.get_reconstructed_regions_with_radius(radius)
        genes_left = region_left.get_contained_genes()
        genes_right = region_right.get_contained_genes()
        genes_left = list(set([g.gene_name for g in genes_left]).intersection(key_genes_set))
        genes_right = list(set([g.gene_name for g in genes_right]).intersection(key_genes_set))
        return genes_left, genes_right

    @staticmethod
    def from_breakfinder_row(row: Any) -> Breakpoint:
        """Return a breakpoint object from a row of a breakfinder file.

        Expects each row to have columns:

            chr1 x1 x2 chr2 y1 y2 strand1 strand2 resolution -logP

        """
        strandA = Strand.from_string(row["strand1"])
        strandB = Strand.from_string(row["strand2"])
        
        chromA = row["chr1"]
        startA = row["x1"]
        endA = row["x2"]
        break_regionA = GenomicRegion(chromA, startA, endA)
        posA = row["x2"] if strandA == Strand.POS else row["x1"]

        chromB = row["chr2"]
        startB = row["y1"]
        endB = row["y2"]
        break_regionB = GenomicRegion(chromB, startB, endB)
        posB = row["y2"] if strandB == Strand.POS else row["y1"]
        
        resolution = resolution_to_int(row["resolution"])
        neg_log_pval = row["-logP"]
        return Breakpoint(chromA, posA, strandA, chromB, posB, strandB, resolution, neg_log_pval, break_regionA, break_regionB)

    @staticmethod
    def list_from_breakfinder_file(filepath: str | StringIO, skiprows: int | None=None) -> list[Breakpoint]:
        """Reads hic_breakfinder output file and returns a sorted list of Breakpoint objects.
    
        This function is designed to work on the Arima-SV Pipeline breaks.bedpe file,
        which is usually located at `output/hic_breakfinder/{id}.breaks.bedpe`.
        If skiprows is None (default), the first row will be skipped if it starts with #.
        """
    
        # Peek at first row
        # The first row of the .breaks.bedpe file is usually a comment with the column names, so skip it
        if skiprows is None:
            if isinstance(filepath, StringIO):
                skiprows = 1 if filepath.read(1) == "#" else 0
            else:
                with open(filepath) as f:
                    skiprows = 1 if f.read(1) == "#" else 0

        df = pd.read_csv(filepath, sep="\s+", names=BREAKFINDER_COLUMNS, skiprows=skiprows)
        
        # Read breakpoints
        breakpoints = [Breakpoint.from_breakfinder_row(row) for _, row in df.iterrows()]
    
        # Sort
        breakpoints.sort()

        return breakpoints

    @staticmethod
    def from_eaglec_row(row: Any) -> Breakpoint:
        """Reads eaglec output file and returns a sorted list of breakpoints.
        
        Expects each row to have columns:
        
            chrA chrB strands posA posB category
            
        """
        chromA = row["chrA"]
        chromB = row["chrB"]
        strands = row["strands"]
        strandA = Strand.from_string(strands[0])
        strandB = Strand.from_string(strands[1])
        posA = row["posA"]
        posB = row["posB"]
        return Breakpoint(chromA, posA, strandA, chromB, posB, strandB)


@dataclass
class QCData:
    """Quality control metrics, designed to work with the Arima-SV pipeline QC outputs.

    This can be reviewed later for more generic QC metrics.

    Properties:
    - raw_pairs: int
    - mapped_se_reads: int
    - mapped_se_reads_pct: float
    - unique_valid_pairs: int
    - unique_valid_pairs_pct: float
    - intra_pairs: int
    - intra_pairs_pct: float
    - intra_ge_15kb_pairs: int
    - intra_ge_15kb_pairs_pct: float
    - inter_pairs: int
    - inter_pairs_pct: float
    - truncated_pct: float
    - duplicated_pct: float
    - invalid_pct: float
    - same_circular_pct: float
    - same_dangling_pct: float
    - same_fragment_internal_pct: float
    - re_ligation_pct: float
    - contiguous_pct: float
    - wrong_size_pct: float
    - mean_lib_length: int
    - lcis_trans_ratio: float
    - num_breakfinder_calls: int    
    """

    raw_pairs: int
    mapped_se_reads: int
    mapped_se_reads_pct: float
    unique_valid_pairs: int
    unique_valid_pairs_pct: float
    intra_pairs: int
    intra_pairs_pct: float
    intra_ge_15kb_pairs: int
    intra_ge_15kb_pairs_pct: float
    inter_pairs: int
    inter_pairs_pct: float
    truncated_pct: float
    duplicated_pct: float
    invalid_pct: float
    same_circular_pct: float
    same_dangling_pct: float
    same_fragment_internal_pct: float
    re_ligation_pct: float
    contiguous_pct: float
    wrong_size_pct: float
    mean_lib_length: int
    lcis_trans_ratio: float
    num_breakfinder_calls: int

    def get_uvp(self) -> int:
        """Return unique valid pairs"""
        return self.unique_valid_pairs

    def get_uvp_pct_raw(self) -> float:
        """Returns the number of UVPs divided by the total number of raw pairs.
        Note: may differ from "unique_valid_pairs_pct" as this appears to be out of aligned pairs. 
        """
        return self.unique_valid_pairs / self.raw_pairs

    @staticmethod
    def from_file(filepath: str):
        """Read a qc_file and return a QCData object.
    
        This function is designed to work on the Arima-SV Pipeline deep QC file,
        which is usually located at `output/{id}_v1.3_Arima_QC_deep.txt`.
    
        """
    
        # Read the Arima-SV Pipeline QC deep file
        # The file format is TSV with a single header row
        df = pd.read_csv(filepath, sep="\s+", header=0)
    
        # The QC deep file is essentially a table with a single row - get the row
        row = df.iloc[0]
    
        # Return a QCData object
        return QCData(
            raw_pairs=row["Raw_pairs"],
            mapped_se_reads=row["Mapped_SE_reads"],
            mapped_se_reads_pct=row["%_Mapped_SE_reads"],
            unique_valid_pairs=row["Unique_valid_pairs"],
            unique_valid_pairs_pct=row["%_Unique_valid_pairs"],
            intra_pairs=row["Intra_pairs"],
            intra_pairs_pct=row["%_Intra_pairs"],
            intra_ge_15kb_pairs=row["Intra_ge_15kb_pairs"],
            intra_ge_15kb_pairs_pct=row["%_Intra_ge_15kb_pairs"],
            inter_pairs=row["Inter_pairs"],
            inter_pairs_pct=row["%_Inter_pairs"],
            truncated_pct=row["%_Truncated"],
            duplicated_pct=row["%_Duplicated_pairs"],
            invalid_pct=row["%_Invalid_pairs"],
            same_circular_pct=row["%_Same_circularised_pairs"],
            same_dangling_pct=row["%_Same_dangling_ends_pairs"],
            same_fragment_internal_pct=row["%_Same_fragment_internal_pairs"],
            re_ligation_pct=row["%_Re_ligation_pairs"],
            contiguous_pct=row["%_Contiguous_sequence_pairs"],
            wrong_size_pct=row["%_Wrong_size_pairs"],
            mean_lib_length=row["Mean_lib_length"],
            lcis_trans_ratio=row["Lcis_trans_ratio"],
            num_breakfinder_calls=row["SVs"],
        )

@dataclass
class ArimaPipelineSample:
    """A sample that was run through the Arima-SV Pipeline.

    Any sample that has been run through the Arima-SV Pipeline will have:
    1. An ID / sample name
    2. QC metrics ("output/{id}_v1.3_Arima_QC_deep.txt")
    3. A .hic file ("output/{id}_inter_30.hic")
    4. Breakpoint calls ("output/hic_breakfinder/{id}.breaks.bedpe", or an alternative curated set)

    For flexibility, any of these attributes (except from the .hic file) can be None
    if the data is not available (e.g. if just the .hic matrices are being shared).

    A normalization constant is calculated and stored for each sample, based on the
    sum of all data in the matrix. 
    
    This normalization is purely used to compare different Hi-C samples
    (i.e. it has no significance if you're just looking at a single sample). 

    Properties:
    - id: str
    - hic: HiCFile object
    - qc: QCData | None
    - breakpoints: list[Breakpoint] | None
    - norm_constant: float (the sum of all raw intra_chromosomal matrix values at 2.5Mb resolution, divided by 1e6)

    """

    id: str
    hic: HiCFile
    qc: QCData | None
    breakpoints: list[Breakpoint] | None
    norm_constant: float

    def __init__(self, id: str, hic_path: str, qc_path: str | None, breakpoints_path: str | None):
        """Initialize ArimaPipelineSample from paths i.e. ArimaPipelineSample(id, hic_path, qc_path, breakpoints_path).
        
        Calculates normalization constant on initialization. 
        
        """
        
        self.id = id
        self.hic = read_hic(hic_path)
        self.qc = QCData.from_file(qc_path) if qc_path is not None else None
        self.breakpoints = Breakpoint.list_from_breakfinder_file(breakpoints_path) if breakpoints_path is not None else None

        norm_constant = 0.0
        for chrom in CHROMS:
            zoom_data = self.hic.getMatrixZoomData(chr_unprefix(chrom), chr_unprefix(chrom), "observed", "NONE", "BP", 2500000)
            data = zoom_data.getRecordsAsMatrix(0, CHROM_SIZES[chrom], 0, CHROM_SIZES[chrom])
            norm_constant += data.sum()
        norm_constant /= 1e6 # expressed in millions of reads
        self.norm_constant = norm_constant

    def __str__(self) -> str:
        """Formatted as id (num breakpoints) (norm norm_constant)"""
        breakpoints_string = "" if self.breakpoints is None else f" ({len(self.breakpoints)} breakpoints)"
        return f"{self.id}{breakpoints_string} (norm {self.norm_constant:.1f})"

    def __repr__(self) -> str:
        """Formatted as ArimaPipelineSample(id=id, hic=HiCFile, qc=[uvp=XM], breakpoints=[n breakpoints], norm_constant=x)"""
        qc_string = "None" if self.qc is None else f"[uvp={self.qc.unique_valid_pairs/1e6:.1f}M]"
        breakpoints_string = "None" if self.breakpoints is None else f" [{len(self.breakpoints)} breakpoints]"
        return f"ArimaPipelineSample(id={self.id}, hic=HiCFile, qc={qc_string}, breakpoints={breakpoints_string}, norm_constant={self.norm_constant})"
        
    def get_hic_region_data(
        self,
        regionX: GenomicRegion,
        regionY: GenomicRegion,
        resolution: int,
        measure: str="observed",
        normalization="NONE",
        norm_constant_normalize=False,
    ) -> NDArray:
        """Get HiC matrix data in a given range at a given resolution as numpy array.
    
        This wrapper is necessary to handle a few quirks of the HiCFile class:
    
        1. The chromosomes in the .hic files are unprefixed, so HiCFile methods require unprefixed chromosome names.
           This wrapper therefore takes care of this conversion: supplied regions contain prefixed chromosome names.
        2. HiCStraw returns a (1,1) matrix if there's no data in the region, so the shape must be adjusted if this occurs.
           (this function will ensure the data is always returned in a shape consistent with the region)
        3. The data retrieval bounds must be constrained to (0, chrom_size).

        If the resulting data is smaller than expected (i.e. at the ends of a chromosome),
        the shape is adjusted to match the expected region size with negative floats as a fill (for plotting with gray masking).
    
        This returns a numpy array with regionX on the x axis and regionY on the y axis.
        (the default from hicstraw is the first region on the y axis and the second region on the x axis,
        so the data is transposed on retrieval from hicstraw).

        The supplied regionX and regionY MUST be bin-aligned, and they must be aligned to a multiple of resolution.
        This will retrieve data from the start bin starting at start, to the bin ending just before end (ending at end-1).

        Measure can be "observed", "oe", ("observed" by default). 
    
        Normalization can be "NONE", "VC", "VC_SQRT" or "KR" ("NONE" by default).

        norm_constant_normalize=True will divide the entire data output by the norm_constant. 

        If a region(s) is tagged with reverse, the matrix will be flipped accordingly. 
        """
    
        # Unpack the regions for convenience
        # chrX and chrY refer to the chromosomes on the X and Y axis, NOT chromosomes X and Y!
        chrX, startX, endX = regionX.get_unpacked()
        chrY, startY, endY = regionY.get_unpacked()
    
        # REQUIRE that regionX and regionY are bin aligned to a multiple of resolution
        assert regionX.bin_align_res is not None and regionX.bin_align_res % resolution == 0
        assert regionY.bin_align_res is not None and regionY.bin_align_res % resolution == 0
    
        # Calculate the expected size of the final hic data matrix
        # The true extents are guaranteed to align to a bin
        expX = (endX - startX) // resolution
        expY = (endY - startY) // resolution
    
        # Next, get the actual .hic data.
        # hicstraw only accepts first arg chr <= second arg chr, so we need to switch chromosomes and transpose if needed.
        # Also need to use the unprefixed chromosome names for hicstraw.
        # Also guard the bounds of the data retrieval to be within 0 and the chromosome sizes.
        # Retrieve one less than the end to prevent retrieving an extra bin if bin falls on bin border
        if regionY < regionX:
            # Retrieves chrY on 1st axis (y axis) and chrX on 2nd axis (x axis), so no need to transpose
            zoom_data = self.hic.getMatrixZoomData(
                chr_unprefix(chrY),
                chr_unprefix(chrX),
                measure,
                normalization,
                "BP",
                resolution,
            )
            data = zoom_data.getRecordsAsMatrix(
                max(0, startY),
                min(endY - 1, CHROM_SIZES[chrY]),
                max(0, startX),
                min(endX - 1, CHROM_SIZES[chrX]),
            )
        else:
            # Retrieves chrX on 1st axis (y axis) and chrY on 2nd axis (x axis) so need to transpose after
            zoom_data = self.hic.getMatrixZoomData(
                chr_unprefix(chrX),
                chr_unprefix(chrY),
                measure,
                normalization,
                "BP",
                resolution,
            )
            data = zoom_data.getRecordsAsMatrix(
                max(0, startX),
                min(endX - 1, CHROM_SIZES[chrX]),
                max(0, startY),
                min(endY - 1, CHROM_SIZES[chrY]),
            )
            # Data was retrieved as (x, y) - transpose to (y, x) for consistency
            data = data.T
    
        # Now we have a data matrix
        # If (1,1) matrix from read hic, then it is 0 data - give expected shape (within bounds) and fill with zero
        if data.shape == (1, 1):
            boundedY = (
                min(endY, CHROM_SIZES[chrY] + resolution)
                - (max(0, startY) // resolution * resolution)
            ) // resolution
            boundedX = (
                min(endX, CHROM_SIZES[chrX] + resolution)
                - (max(0, startX) // resolution * resolution)
            ) // resolution
            # Some calls may have been given out of chromosome bounds anyway: guard against this
            boundedY = max(boundedY, 1)
            boundedX = max(boundedX, 1)
            data = np.zeros((boundedY, boundedX))
    
        # If the data shape is not as expected, then the provided range is probably at a boundary.
        # Therefore, bring to correct shape and fill boundaries with negative values (which will be grey-masked later)
        # Fill in missing data on Y-axis (axis 0)
        if data.shape[0] < expY:
            # If filler is needed at both ends, calculate the amount
            if startY < 0 and endY > CHROM_SIZES[chrY]:
                yStartExcess = (0 - startY) // resolution
                yEndExcess = (endY - CHROM_SIZES[chrY]) // resolution
                fillerYStart = np.zeros((yStartExcess, data.shape[1])) - 1
                fillerYEnd = np.zeros((yEndExcess, data.shape[1])) - 1
                data = np.vstack([fillerYStart, data, fillerYEnd])
            else:
                filler = np.zeros((expY - data.shape[0], data.shape[1])) - 1
                # Prepend the filler if the start was less than 0, otherwise append after end
                if startY < 0:
                    data = np.vstack([filler, data])
                else:
                    data = np.vstack([data, filler])
        # Fill in missing data on X-axis (axis 1)
        if data.shape[1] < expX:
            if startX < 0 and endX > CHROM_SIZES[chrX]:
                xStartExcess = (0 - startX) // resolution
                xEndExcess = (endX - CHROM_SIZES[chrX]) // resolution
                fillerXStart = np.zeros((data.shape[0], xStartExcess)) - 1
                fillerXEnd = np.zeros((data.shape[0], xEndExcess)) - 1
                data = np.hstack([fillerXStart, data, fillerXEnd])
            else:
                filler = np.zeros((data.shape[0], expX - data.shape[1])) - 1
                if startX < 0:
                    data = np.hstack([filler, data])
                else:
                    data = np.hstack([data, filler])
    
        # Ensure shape matches expected shape
        assert data.shape[0] == expY and data.shape[1] == expX
    
        # Divide by normalization constant
        if norm_constant_normalize:
            data = data / self.norm_constant
            
        # Mask values below -1 (fill values)
        masked = np.ma.masked_where(data < 0, data)

        # Flip if region is reversed
        if regionX.reverse:
            masked = np.flip(masked, axis=1)
        if regionY.reverse:
            masked = np.flip(masked, axis=0)
    
        return masked
    
    
    def get_hic_direct_data(
        self,
        regionX: GenomicRegion,
        regionY: GenomicRegion,
        resolution: int,
        measure: str="observed",
        normalization="NONE",
        norm_constant_normalize=False,
    ) -> NDArray:
        """Gets hic data direct from hicstraw (less convenient than the region wrapper above, but useful for whole-chromosome plots).

        This will NOT perform any checking of regionX/regionY bounds, and will not perform any zero-padding. 
        
        """
        
        chrX, startX, endX = regionX.get_unpacked()
        chrY, startY, endY = regionY.get_unpacked()
        
        if regionY < regionX:
            zoom_data = self.hic.getMatrixZoomData(
                chr_unprefix(chrY),
                chr_unprefix(chrX),
                measure,
                normalization,
                "BP",
                resolution,
            )
            data = zoom_data.getRecordsAsMatrix(
                max(0, startY),
                min(endY, CHROM_SIZES[chrY]),
                max(0, startX),
                min(endX, CHROM_SIZES[chrX]),
            )
        else:
            zoom_data = self.hic.getMatrixZoomData(
                chr_unprefix(chrX),
                chr_unprefix(chrY),
                measure, 
                normalization,
                "BP",
                resolution,
            )
            data = zoom_data.getRecordsAsMatrix(
                max(0, startX),
                min(endX, CHROM_SIZES[chrX]),
                max(0, startY),
                min(endY, CHROM_SIZES[chrY]),
            )
            data = data.T
    
        # Divide by normalization constant
        if norm_constant_normalize:
            data = data / self.norm_constant

        # Flip if region is reversed
        if regionX.reverse:
            data = np.flip(data, axis=1)
        if regionY.reverse:
            data = np.flip(data, axis=0)
            
        return data

    def get_assembled_hic(
        self, 
        assembly: list[GenomicRegion], 
        resolution: int, 
        norm: str="NONE", 
        measure: str="observed", 
        norm_constant_normalize=False, 
        gap_size: int = 0, 
        gap_value=np.nan
    ) -> AssembledHic:
        """Return an AssembledHic object for a assembled data over the given assembly regions.

        Takes an assembly (ordered sequential list of genomic regions) and retrieves and reconstructs the Hi-C data. 

        Every region MUST be a multiple of the resolution. 

        Gaps are filled with gap_value (default np.nan) in the data matrix. 
        Default gap size is 0 (i.e. no gaps). 
        """

        for gregion in assembly:
            assert gregion.start % resolution == 0
            assert gregion.end % resolution == 0
    
        # Get matrix data for each pair of segments; assemble row by row 
        rows = [] 
        for y, segmentY in enumerate(assembly):
            row = []
            row_divisions: list[tuple[GenomicRegion | None, tuple[int, int]]]  = []
            col_pointer = 0 
            for x, segmentX in enumerate(assembly):
                pair_data = self.get_hic_region_data(segmentX, segmentY, resolution, measure, norm, norm_constant_normalize)
                row.append(pair_data)
                row_divisions.append((segmentX, (col_pointer, col_pointer + pair_data.shape[1])))
                col_pointer += pair_data.shape[1]
                if x < len(assembly)-1 and gap_size > 0:
                    gap_matrix = np.empty((pair_data.shape[0], gap_size))
                    gap_matrix[:] = gap_value
                    row.append(gap_matrix)
                    row_divisions.append((None, (col_pointer, col_pointer + gap_size)))
                    col_pointer += gap_size
            stacked = np.hstack(row)
            rows.append(stacked)
            if y < len(assembly)-1 and gap_size > 0:
                gap_matrix = np.empty((gap_size, stacked.shape[1]))
                gap_matrix[:] = gap_value
                rows.append(gap_matrix)
        assembled = np.vstack(rows)
    
        plot_regions = [ 
            PlotRegion(
                bin_range = (start, end),
                genomic_region = seg
            )
            for seg, (start, end) in row_divisions
        ]
    
        return AssembledHic(
            data = assembled,
            plot_regions = plot_regions
        )

    def get_genome_wide_virtual_4c_at_locus(self, region: GenomicRegion, resolution: int, norm="NONE", measure="oe") -> list[tuple[str, NDArray]]:
        """Return genome-wide Hi-C for a region (virtual 4C), taking the mean of the values in the region width."""
        all_values = []

        chr = region.chrom
        start = region.start
        end = region.end 
        
        # Assemble for each chromosome
        for chr_partner in CHROMS:
            if chr_partner == "chrY":
                continue
            regionX = GenomicRegion(chr_partner, 0, CHROM_SIZES[chr_partner])
            data = self.get_hic_direct_data(regionX, region, resolution=resolution, normalization=norm, measure=measure)
            data = data.mean(axis=0)
            all_values.append((chr_partner, data))

        return all_values

