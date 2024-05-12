"""Command line interface to generator."""

import argparse
from hicdash.generator import make_html_report_and_save


def main():

    parser = argparse.ArgumentParser(description='Generate a Hi-C report from Arima-SV Pipeline outputs.')
    parser.add_argument("--prefix", type=str, help="Prefix of the output files from Arima-SV Pipeline (i.e. the sample ID).")
    parser.add_argument("--qc", type=str, help="Filepath to deep QC .txt file from Arima-SV Pipeline.", default=None)
    parser.add_argument("--hic", type=str, help="Filepath to Hi-C .hic file from Arima-SV Pipeline.")
    parser.add_argument("--breaks", type=str, help="Filepath to hic_breakfinder breaks.bedpe from Arima-SV Pipeline for regional zooms.", default=None)
    parser.add_argument("--bedpe", type=str, help="Filepath(s) (comma-delimited) to extra .bedpe files to be annotated.", default=None)
    parser.add_argument("--bigwig", type=str, help="Filepath(s) (comma-delimited) to extra .bigwig files to be annotated.", default=None)
    parser.add_argument("--control", type=str, help="Filepath to control Hi-C file for visual comparison.", default=None)
    parser.add_argument("--output", type=str, help="Filepath to save the report into (include .html extension).")

    args = parser.parse_args()

    assert args.prefix, "Please provide the prefix (sample ID)."
    assert args.hic.endswith(".hic"), "Please provide a filepath to the Hi-C .hic file."
    assert args.output, "Please provide a filepath to save the report into."

    make_html_report_and_save(
        sample_id=args.prefix,
        qc_filepath=args.qc,
        hic_filepath=args.hic,
        breakfinder_filepath=args.breaks,
        extra_bedpe=args.bedpe,
        extra_bigwig=args.bigwig,
        output_filepath=args.output,
        control_filepath=args.control,
    )


if __name__ == "__main__":
    main()