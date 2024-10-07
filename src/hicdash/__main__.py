"""Command line interface to generator.

Example usage:

    python -m hicdash \
        --prefix K562 \
        --hic tests/example_data/K562_inter_30.hic \
        --output tests/example_data/K562_report.html \
        --qc tests/example_data/K562_v1.3_Arima_QC_deep.txt \
        --breaks tests/example_data/K562_curated_breakpoints.bedpe \
        --control tests/example_data/GM12878_inter_30.hic \
        --keygenes tests/example_data/keygenes.txt  \
        --targets1 tests/example_data/targets1.txt \
        --targets2 tests/example_data/targets2.txt \
        --flagged tests/example_data/flagged.txt

"""

import argparse
import sys

from hicdash.generator import render_sample_report

def main():

    parser = argparse.ArgumentParser(description='Generate a Hi-C report from Arima-SV Pipeline outputs.')
    parser.add_argument("--prefix", type=str, help="Prefix of the output files from Arima-SV Pipeline (i.e. the sample ID).")
    parser.add_argument("--qc", type=str, help="Filepath to deep QC .txt file from Arima-SV Pipeline.", default=None)
    parser.add_argument("--hic", type=str, help="Filepath to Hi-C .hic file from Arima-SV Pipeline.")
    parser.add_argument("--breaks", type=str, help="Filepath to bedpe file containing breakpoints (in hic_breakfinder format).", default=None)
    parser.add_argument("--control", type=str, help="Filepath to control Hi-C file for visual comparison.", default=None)
    parser.add_argument("--output", type=str, help="Filepath to save the report into (include .html extension).")
    parser.add_argument("--keygenes", type=str, help="Filepath to list of key genes (one gene name per line) for annotation.", default=None)
    parser.add_argument("--targets1", type=str, help="Filepath to list of target genes (one gene name per line) for pseudotargeted Hi-C (single-target).", default=None)
    parser.add_argument("--targets2", type=str, help="Filepath to list of target gene pairs (tab-delimited pairs, one pair per line) for pseudotargeted Hi-C (dual-target).", default=None)
    parser.add_argument("--flagged", type=str, help="Filepath to bedpe of flagged common false positive breakpoint regions.", default=None)
    # parser.add_argument("--bedpe", type=str, help="Filepath(s) (comma-delimited) to extra .bedpe files to be annotated.", default=None)
    # parser.add_argument("--bigwig", type=str, help="Filepath(s) (comma-delimited) to extra .bigwig files to be annotated.", default=None)
    # parser.add_argument("--crosshairs", type=bool, help="Add crosshairs to all breaks and annotations.", default=False, action=argparse.BooleanOptionalAction)
    # parser.add_argument("--grid", type=bool, help="Add a grid to the zoomed plot.", default=False, action=argparse.BooleanOptionalAction)

    args = parser.parse_args()

    if args.prefix is None:
        print("Please provide the prefix (sample ID) (--prefix)")
        return 1
    if args.hic is None or not args.hic.endswith(".hic"):
        print("Please provide a filepath to the Hi-C .hic file (--hic)")
    if not args.output or not args.output.endswith(".html"):
        print("Please provide a filepath to save the report into and ensure it ends with the .html extension (--output)")

    render_sample_report(
        prefix=args.prefix,
        hic_filepath=args.hic,
        qc_filepath=args.qc,
        breakpoints_filepath=args.breaks,
        extra_bedpe=None,
        extra_bigwig=None,
        control_filepath=args.control,
        output_filepath=args.output,
        key_genes_filepath=args.keygenes,
        targets1_filepath=args.targets1,
        targets2_filepath=args.targets2,
        command=" ".join(sys.argv),
        flagged_filepath=args.flagged,
    )

if __name__ == "__main__":
    main()