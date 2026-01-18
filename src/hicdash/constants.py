"""Useful package-wide basic constants.

Currently these constants restrict report generation to:
- hg38-aligned Hi-C files
- Ensembl 110 gene annotations

To be reviewed at a later stage. 
"""

import pyensembl # type: ignore

# -------------------------------------------------------------------------------
# HUMAN CHROMOSOMES (hg38)
# -------------------------------------------------------------------------------

# List of human chromosomes (prefixed with "chr")
CHROMS = [
    "chr1",
    "chr2",
    "chr3",
    "chr4",
    "chr5",
    "chr6",
    "chr7",
    "chr8",
    "chr9",
    "chr10",
    "chr11",
    "chr12",
    "chr13",
    "chr14",
    "chr15",
    "chr16",
    "chr17",
    "chr18",
    "chr19",
    "chr20",
    "chr21",
    "chr22",
    "chrX",
    "chrY",
]

# Dictionary of chromosome indices indicating the order of chromosomes
CHROM_INDICES = {chrom: i for i, chrom in enumerate(CHROMS)}

# hg38 chromosome sizes
CHROM_SIZES = {
    "chr1": 248956422,
    "chr2": 242193529,
    "chr3": 198295559,
    "chr4": 190214555,
    "chr5": 181538259,
    "chr6": 170805979,
    "chr7": 159345973,
    "chr8": 145138636,
    "chr9": 138394717,
    "chr10": 133797422,
    "chr11": 135086622,
    "chr12": 133275309,
    "chr13": 114364328,
    "chr14": 107043718,
    "chr15": 101991189,
    "chr16": 90338345,
    "chr17": 83257441,
    "chr18": 80373285,
    "chr19": 58617616,
    "chr20": 64444167,
    "chr21": 46709983,
    "chr22": 50818468,
    "chrX": 156040895,
    "chrY": 57227415,
}

# -------------------------------------------------------------------------------
# GENE ANNOTATIONS
# -------------------------------------------------------------------------------

# Ensembl gene annotations
GENE_ANNOTATIONS = pyensembl.EnsemblRelease(110, "human")


# -------------------------------------------------------------------------------
# COLUMN NAMES
# -------------------------------------------------------------------------------

# Column names for the hic_breakfinder output bedpe file.
BREAKFINDER_COLUMNS = [
    "chr1",
    "x1",
    "x2",
    "chr2",
    "y1",
    "y2",
    "strand1",
    "strand2",
    "resolution",
    "-logP",
]


BEDPE_COLUMNS = [
    "chr1",
    "start1",
    "end1",
    "chr2",
    "start2",
    "end2",
]

# -------------------------------------------------------------------------------
# ANNOTATION COLOURS
# -------------------------------------------------------------------------------

BEDPE_COLORS = [
    "blue",
    "green",
    "magenta",
    "cyan",
    "yellow",
    "orange",
    "purple",
    "lime",
    "indigo",
    "pink",
    "teal",
    "brown",
]

BIGWIG_COLORS = [
    "blue",
    "dodgerblue",
    "teal",
    "green",
    "lime",
    "yellow",
]

MARKER_SIZE_DICT = {
    2500000: 6,
    1000000: 7,
    500000: 8,
    250000: 9,
    100000: 10,
    50000: 13,
    25000: 14,
    10000: 15,
    5000: 17,
    1000: 20,
}

CHROM_COLORS = {
    "chr1": "#B15928",
    "chr2": "#6A3D9A",
    "chr3": "#CAB2D6",
    "chr4": "#FDBF6F",
    "chr5": "#A6CEE3",
    "chr6": "#B2DF8A",
    "chr7": "#1B9E77",
    "chr8": "#A6CEE3",
    "chr9": "#33A02C",
    "chr10": "#E6AB02",
    "chr11": "#E58606",
    "chr12": "#5D69B1",
    "chr13": "#FF7F00",
    "chr14": "#52BCA3",
    "chr15": "#99C945",
    "chr16": "#CC61B0",
    "chr17": "#88CCEE",
    "chr18": "#1D6996",
    "chr19": "#117733",
    "chr20": "#661100",
    "chr21": "#882255",
    "chr22": "#1F78B4",
    "chrX": "#666666",
    "chrY": "#5F4690",
}

# Shorten norm names to a single letter for plots
SHORT_NORM = {
    "NONE": "R",
    "SCALE": "S",
    "VC_SQRT": "V"
}