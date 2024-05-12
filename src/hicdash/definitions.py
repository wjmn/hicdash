"""Useful package-wide basic types.

These types help provide the basis for useful type signatures for 
functions and classes in the package, as well as type hints 
during development. 

"""

from dataclasses import dataclass
from enum import Enum
from hicstraw import HiCFile

# -------------------------------------------------------------------------------
# STRUCTURAL VARIANTS
# -------------------------------------------------------------------------------


class Pairing(Enum):
    """Enum indicating the chromosomal pairing of a structural variant.

    Structural variants can be either:

    - Intra-chromosomal (both breakpoints on the same chromosome), or
    - Inter-chromosomal (each breakpoint on a different chromosome).

    """

    INTER = "inter-chromosomal"
    INTRA = "intra-chromosomal"


class VariantCategory(Enum):
    """Enum indicating the type (e.g. deletion) of a structural variant.

    Simple structural variants can be categorised as either:

    - Deletion
    - Duplication
    - Inversion, or other intra-chromosomal event.
    - Translocation

    Deletions, duplications and inversions are intra-chromosomal events, while
    translocations are inter-chromosomal events.

    The determination of the category of a structural variant is based on the
    strandness of each breakpoint. See:

    Song F, Xu J, Dixon J, Yue F. Analysis of Hi-C Data for Discovery of Structural Variations in Cancer. Methods Mol Biol. 2022;2301:143-161.

    For the purposes of this dashboard generator, structural variants are
    categorised in isolation. The following rules are therefore used:

    1. All inter-chromosomal breakpoints are categorised as translocations.
    2. An intra-chromosomal breakpoint with +/- strandness is a deletion.
    3. An intra-chromosomal breakpoint with -/+ strandness is a duplication.
    4. Any other intra-chromosomal breakpoints are categorised as inversions
       or other.

    This isn't always perfect, so take these categories with a grain of salt. 

    Determining proper inversions requires two structural variant calls from
    the default outputs of hic_breakfinder, so for simplicity to categorise
    all structural variant calls in isolation, they are simply grouped
    here as inversions *or other*.

    """

    DELETION = "deletion"
    DUPLICATION = "duplication"
    INVERSION_OR_OTHER = "inversion-or-other"
    TRANSLOCATION = "translocation"


class Strand(Enum):
    """Enum indicating the strandness of a structural variant breakpoint.

    A strand can be either:
    - Positive, or
    - Negative.

    As per hic_breakfinder documentation, strands indicate predictions of
    which end the true breakpoint is closest to (as hic_breakfinder reports
    breakpoints as ranges rather than a single coordinate).

    - Positive indicates the "end" coordinate is the true breakpoint
    - Negative indicates the "start" coordinate is the true breakpoint

    """

    POS = "+"
    NEG = "-"


def to_strand(s: str) -> Strand:
    """Convert a string to a Strand enum.

    Raises an ValueError if the string is not a valid strand. Only "+" and "-"
    are valid input strings.

    """
    if s == "+":
        return Strand.POS
    elif s == "-":
        return Strand.NEG
    else:
        raise ValueError(f"Invalid strand: {s}")


# -------------------------------------------------------------------------------
# DATA CLASSES
# -------------------------------------------------------------------------------


@dataclass
class QCData:
    """Quality control metrics, designed to work with the Arima-SV pipeline QC outputs.

    This can be reviewed later for more generic QC metrics.
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

@dataclass
class Breakpoint:
    """A breakpoint is a position on a chromosome with strandness.

    hic_breakfinder breakpoints are actually given as ranges, with
    a start and an end.
    
    For convenience, the predicted position of each breakpoint
    (based on strandness - see Strand enum) is included as `pos`.

    Other breakpoint callers - such as EagleC - give only a single position.
    In this case, start, end, and pos will all have the same value. 

    - Positive strand: the "end" coordinate is the predicted breakpoint
    - Negative strand: the "start" coordinate is the predicted breakpoint

    Chromosomes in these objects must be prefixed with "chr". 
    """

    chr: str
    start: int
    end: int
    pos: int
    strand: Strand

@dataclass
class BreakfinderCall:
    """A structural variant call made by hic_breakfinder.

    breakA should be guaranteed to be the breakpoint with the lower chromosome index.

    Resolution indicates the resolution the call was made at. 

    """

    breakpointA: Breakpoint
    breakpointB: Breakpoint
    resolution: int
    neg_log_pval: float
    pairing: Pairing
    category: VariantCategory

@dataclass
class BedpeLine: 
    """A generic bedpe line, with 6 guaranteed fields only."""

    chrA: str
    startA: int
    endA: int
    chrB: str
    startB: int
    endB: int

@dataclass
class ArimaPipelineSample:
    """A sample that was run through the Arima-SV Pipeline.

    Any sample that has been run through the Arima-SV Pipeline will have:
    1. An ID / sample name
    2. QC metrics ("output/{id}_v1.3_Arima_QC_deep.txt")
    3. A .hic file ("output/{id}_inter_30.hic")
    4. Breakfinder calls ("output/hic_breakfinder/{id}.breaks.bedpe")

    For flexibility, any of these attributes (except from the .hic file) can be None
    if the data is not available (e.g. if just the .hic matrices are being shared).

    A normalization constant is calculated and stored for each sample, based on a 
    sample of the Hi-C matrix. Ideally this would be based on unique_valid_pairs
    from the QC instead, but because the QC data is not always available, 
    the normalization constant is based on the actual Hi-C data. 
    
    This normalization is purely used to compare different Hi-C samples
    (i.e. it has no significance if you're just looking at a single sample). 

    """

    id: str
    hic: HiCFile
    qc: QCData | None
    breakfinder_calls: list[BreakfinderCall] | list[BedpeLine] | None
    norm_constant: float


@dataclass
class Region: 
    """A genomic region with a chromosome, start and end position.

    Chromosomes should be prefixed with "chr".

    Breakpoints are actually regions as well (with additional fields).
    """

    chr: str
    start: int
    end: int