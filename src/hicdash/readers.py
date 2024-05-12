"""Functions for reading data and converting to types defined in types.py.
"""

import pandas as pd
from hicstraw import HiCFile
from hicdash.constants import CHROMS, CHROM_INDICES, CHROM_SIZES
from hicdash.definitions import (
    to_strand,
    QCData,
    BreakfinderCall,
    ArimaPipelineSample,
    Pairing,
    VariantCategory,
    Strand,
    Breakpoint,
)
from hicdash.utilities import chr_prefix, chr_unprefix, to_mega, resolution_to_int


def read_qc(qc_filepath: str) -> QCData:
    """Read a qc_file and return a QCData object.

    This function is designed to work on the Arima-SV Pipeline deep QC file,
    which is usually located at `output/{id}_v1.3_Arima_QC_deep.txt`.

    """

    # Read the Arima-SV Pipeline QC deep file
    # The file format is TSV with a single header row
    df = pd.read_csv(qc_filepath, sep="\t", header=0)

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


def read_breakfinder_data(breakfinder_filepath: str) -> list[BreakfinderCall]:
    """Reads hic_breakfinder output file and returns a list of BreakfinderCall objects.

    This function is designed to work on the Arima-SV Pipeline breaks.bedpe file,
    which is usually located at `output/hic_breakfinder/{id}.breaks.bedpe`.

    For consistency, breakfinder calls here will be processed such that:
    1. All chromosomes are prefixed with "chr" (if not already)
    2. The breakpoint with the lower chromosome index is always breakpointA

    """

    # The first row of the .breaks.bedpe file is a comment with the column names, so skip it
    df = pd.read_csv(
        breakfinder_filepath, sep="\t", names=BREAKFINDER_COLUMNS, skiprows=1
    )

    # Create an empty list, which will be populated with BreakfinderCall objects
    calls = []

    # Iterate through all breakfinder calls (as rows in the dataframe)
    for _, row in df.iterrows():

        # Get data out of the row for processing and make breakpoint objects
        chr1 = chr_prefix(row["chr1"])
        start1 = row["x1"]
        end1 = row["x2"]
        strand1 = to_strand(row["strand1"])
        pos1 = end1 if strand1 == Strand.POS else start1
        breakpointA = Breakpoint(chr1, start1, end1, pos1, strand1)

        chr2 = chr_prefix(row["chr2"])
        start2 = row["y1"]
        end2 = row["y2"]
        strand2 = to_strand(row["strand2"])
        pos2 = end2 if strand2 == Strand.POS else start2
        breakpointB = Breakpoint(chr2, start2, end2, pos2, strand2)

        resolution = resolution_to_int(row["resolution"])
        neg_log_pval = row["-logP"]

        # Ensure breakpointA has the chromosome with the lower index
        # If it's got a greater index, then swap breakpoints
        if CHROM_INDICES[breakpointA.chr] > CHROM_INDICES[breakpointB.chr]:
            breakpointA, breakpointB = breakpointB, breakpointA

        # Assign the call a variant category
        if breakpointA.chr != breakpointB.chr:
            category = VariantCategory.TRANSLOCATION
        else:
            if breakpointA.strand == Strand.POS and breakpointB.strand == Strand.NEG:
                category = VariantCategory.DELETION
            elif breakpointA.strand == Strand.NEG and breakpointB.strand == Strand.POS:
                category = VariantCategory.DUPLICATION
            else:
                category = VariantCategory.INVERSION_OR_OTHER

        # Assign the call a chromosomal pairing
        pairing = Pairing.INTRA if breakpointA.chr == breakpointB.chr else Pairing.INTER

        # Create a BreakfinderCall object
        call = BreakfinderCall(
            breakpointA=breakpointA,
            breakpointB=breakpointB,
            resolution=resolution,
            neg_log_pval=neg_log_pval,
            pairing=pairing,
            category=category,
        )

        # Add to the cumulative list of calls
        calls.append(call)

    # Sort breakfinder calls by chrA region 
    calls.sort(key=lambda x: CHROM_INDICES[x.breakpointA.chr] * 1e11 + CHROM_INDICES[x.breakpointB.chr] * 1e9 + x.breakpointA.pos)

    return calls


def read_hic(hic_file: str) -> HiCFile:
    """Read Hi-C file and return a HiCFile object.
    
    This is just a convenience wrapper for now, to be revised. 
    """
    return HiCFile(hic_file)


def read_sample(id: str, hic_filepath: str, qc_filepath: str | None=None, breakfinder_filepath: str | None=None) -> ArimaPipelineSample:
    """Read a sample's data and return an ArimaPipelineSample object.
    
    Brings all the data together into one object. 

    Calculates the norm constant - ideally this would be based on the unique_valid_pairs from the QC data,
    but because QC data may not always be available, instead calculate the norm constant based on a 
    sample of the Hi-C matrix data (which cannot be None). 

    """

    hic = read_hic(hic_filepath)
    qc = read_qc(qc_filepath) if qc_filepath is not None else None
    breakfinder_calls = read_breakfinder_data(breakfinder_filepath) if breakfinder_filepath is not None else None

    # Calculate the norm constant based on the Hi-C data
    # Currently based on a subsample (chr1), but at some stage maybe try to choose a less arbitrary sample
    zoom_data = hic.getMatrixZoomData("1", "1", "observed", "NONE", "BP", 2500000)
    data = zoom_data.getRecordsAsMatrix(0, CHROM_SIZES["chr1"], 0, CHROM_SIZES["chr1"])
    norm_constant = data.sum() / 1e6

    sample = ArimaPipelineSample(id=id, hic=hic, qc=qc, breakfinder_calls=breakfinder_calls, norm_constant=norm_constant)

    return sample
