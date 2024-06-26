"""Functions for generating Hi-C dashboard / report. 

Most of this is messy HTML templating...to review ongoing.

"""

from hicdash.constants import GENE_ANNOTATIONS, BEDPE_COLORS
from hicdash.utilities import chr_unprefix
from hicdash.definitions import BreakfinderCall, Strand, ArimaPipelineSample, BedpeLine
from hicdash.plotters import (
    plot_composite_double_whole_matrix,
    plot_composite_compare_two,
    plot_composite_context_and_zoom,
    plot_composite_multires_breakpoint,
    plot_qc,
)
from hicdash.readers import read_sample, read_bedpe
from pathlib import Path
import matplotlib.pyplot as plt
import base64
import io
import datetime
import os
import pyBigWig

# -------------------------------------------------------------------------------
# EXTRA UTILITIES
# -------------------------------------------------------------------------------


def get_genes_at_call(call: BreakfinderCall | BedpeLine) -> tuple[list[str], list[str]]:
    """Get a list of genes directly at a given breakfinder call (as (genesA, genesB))"""

    # Unpack breakfinder call
    if isinstance(call, BreakfinderCall):
        chrA, posA = call.breakpointA.chr, call.breakpointA.pos
        chrB, posB = call.breakpointB.chr, call.breakpointB.pos
    elif isinstance(call, BedpeLine):
        chrA, posA = call.chrA, (call.startA + call.endA) // 2
        chrB, posB = call.chrB, (call.startB + call.endB) // 2

    # Get all non-empty genes at call
    non_empty = lambda x: x.gene_name != ""
    genesA = filter(
        non_empty,
        GENE_ANNOTATIONS.genes_at_locus(contig=chr_unprefix(chrA), position=posA),
    )
    genesB = filter(
        non_empty,
        GENE_ANNOTATIONS.genes_at_locus(contig=chr_unprefix(chrB), position=posB),
    )

    return ([g.gene_name for g in genesA], [g.gene_name for g in genesB])


def get_genes_around_call(
    call: BreakfinderCall | BedpeLine, width=300000, buffer=25000, protein_coding=True
):
    """Gets a list of genes around a breakpoint (based on strandness).

    Looks in direction of strandness for WIDTH bp and in opposite direction of strandness for BUFFER bp.
    """

    # Unpack breakfinder call
    if isinstance(call, BreakfinderCall):
        chrA, posA, strandA = (
            call.breakpointA.chr,
            call.breakpointA.pos,
            call.breakpointA.strand,
        )
        chrB, posB, strandB = (
            call.breakpointB.chr,
            call.breakpointB.pos,
            call.breakpointB.strand,
        )

        if strandA == Strand.POS:
            startA = posA - width
            endA = posA + buffer
        else:
            startA = posA - buffer
            endA = posA + width

        if strandB == Strand.POS:
            startB = posB - width
            endB = posB + buffer
        else:
            startB = posB - buffer
            endB = posB + width

    elif isinstance(call, BedpeLine):
        chrA, posA = call.chrA, (call.startA + call.endA) // 2
        chrB, posB = call.chrB, (call.startB + call.endB) // 2

        startA = posA - width // 2
        endA = posA + width // 2
        startB = posB - width // 2
        endB = posB + width // 2

    # Get start and end ranges based on strandness
    non_empty = lambda x: x.gene_name != ""
    genesA = filter(
        non_empty,
        GENE_ANNOTATIONS.genes_at_locus(
            contig=chr_unprefix(chrA), position=startA, end=endA
        ),
    )
    genesB = filter(
        non_empty,
        GENE_ANNOTATIONS.genes_at_locus(
            contig=chr_unprefix(chrB), position=startB, end=endB
        ),
    )

    if protein_coding:
        genesA = [gene for gene in genesA if gene.biotype == "protein_coding"]
        genesB = [gene for gene in genesB if gene.biotype == "protein_coding"]

    # Sort genes by proximity to breakpoint
    genesA.sort(key=lambda x: min(abs(x.end - posA), abs(x.start - posA)))
    genesB.sort(key=lambda x: min(abs(x.end - posB), abs(x.start - posB)))

    return ([g.gene_name for g in genesA], [g.gene_name for g in genesB])


def call_to_string(call: BreakfinderCall | BedpeLine) -> str:
    """Convert a breakfinder call to a string representation"""

    # Unpack call
    if isinstance(call, BreakfinderCall):
        chrA, posA, strandA = (
            call.breakpointA.chr,
            call.breakpointA.pos,
            call.breakpointA.strand,
        )
        chrB, posB, strandB = (
            call.breakpointB.chr,
            call.breakpointB.pos,
            call.breakpointB.strand,
        )

        # Convert to string
        return f"{chrA}:{posA};{chrB}:{posB} ({strandA.value}{strandB.value})"
    elif isinstance(call, BedpeLine):
        return f"{call.chrA}:{call.startA};{call.chrB}:{call.startB}"


def call_to_alphanumeric_string(call: BreakfinderCall) -> str:
    """Convert a breakfinder call to an alphanumeric string representation"""

    # Unpack call
    if isinstance(call, BreakfinderCall):
        chrA, posA, strandA = (
            call.breakpointA.chr,
            call.breakpointA.pos,
            call.breakpointA.strand,
        )
        chrB, posB, strandB = (
            call.breakpointB.chr,
            call.breakpointB.pos,
            call.breakpointB.strand,
        )

        strandAstr = "P" if strandA == Strand.POS else "N"
        strandBstr = "P" if strandB == Strand.POS else "N"

        # Convert to string
        return f"{chrA}-{posA}-{chrB}-{posB}-{strandAstr}{strandBstr}"
    elif isinstance(call, BedpeLine):
        return f"{call.chrA}-{call.startA}-{call.chrB}-{call.startB}"


def fig_to_base64_and_close(fig: plt.Figure) -> str:
    """Converts a matplotlib figure to a base64 string"""

    # Save figure to bytes
    fig_io_bytes = io.BytesIO()
    plt.savefig(fig_io_bytes, format="png", bbox_inches="tight")
    fig_io_bytes.seek(0)
    fig_hash = base64.b64encode(fig_io_bytes.read())

    plt.close(fig)

    return fig_hash.decode("utf-8")


# -------------------------------------------------------------------------------
# TEMPLATE REPLACEMENT
# -------------------------------------------------------------------------------

# First, define the templates as constant strings.
TEMPLATE_DIR = Path(__file__).parent
TEMPLATE_REPORT = Path(TEMPLATE_DIR / "templates" / "report.html").read_text()
TEMPLATE_CALL = Path(TEMPLATE_DIR / "templates" / "call.html").read_text()
TEMPLATE_GENE = Path(TEMPLATE_DIR / "templates" / "gene.html").read_text()
TEMPLATE_QC_PLOT = Path(TEMPLATE_DIR / "templates" / "qc_plot.html").read_text()
TEMPLATE_COMPARISON_PLOT = Path(
    TEMPLATE_DIR / "templates" / "comparison_plot.html"
).read_text()
TEMPLATE_SIDEBAR_CALL = Path(
    TEMPLATE_DIR / "templates" / "sidebar_call.html"
).read_text()


def make_html_sidebar_call(call: BreakfinderCall) -> str:
    """Generate HTML for a call entry in the sidebar (for the report)"""

    # Replace template
    return TEMPLATE_SIDEBAR_CALL.format(
        call_region_string=call_to_string(call),
        call_region_string_no_spaces=call_to_alphanumeric_string(call),
    )


def make_html_gene(gene: str, direct=False) -> str:
    direct = "direct" if direct else "nearby"
    return TEMPLATE_GENE.format(
        gene=gene,
        direct=direct,
    )


def make_html_comparison_plot(
    sample: ArimaPipelineSample,
    control: None | ArimaPipelineSample,
    call: BreakfinderCall | BedpeLine,
) -> str:
    """Generate HTML for a comparison plot entry in the report"""

    # Generate comparison plot
    if control is None:
        return ""
    else:
        comparison_plot_fig = plot_composite_compare_two(sample, control, call)
        comparison_plot_base64 = fig_to_base64_and_close(comparison_plot_fig)

        return TEMPLATE_COMPARISON_PLOT.format(
            comparison_base64=comparison_plot_base64, sample_id=sample.id
        )


def make_html_qc_plot(sample: ArimaPipelineSample) -> str:
    """Generate HTML for a QC plot entry in the report"""

    # Generate QC plot
    if sample.qc is None:
        return "<p>No QC file was provided.</p>"
    else:
        qc_plot_fig = plot_qc(sample)
        qc_base64 = fig_to_base64_and_close(qc_plot_fig)

        return TEMPLATE_QC_PLOT.format(
            qc_base64=qc_base64,
        )


def make_html_call(
    sample: ArimaPipelineSample,
    call: BreakfinderCall | BedpeLine,
    control: ArimaPipelineSample | None = None,
    extra_bedpe_data: list[BedpeLine] = [],
    extra_bigwig_handles: list[tuple[str, object]] = [],
    crosshairs: bool = False,
    grid: bool = False,
) -> str:
    """Generate HTML for a call entry in the report"""

    print(f"- Generating HTML for breakfinder call: {call_to_string(call)}")

    # Unpack call
    if isinstance(call, BreakfinderCall):
        chrA, posA = call.breakpointA.chr, call.breakpointA.pos
        chrB, posB = call.breakpointB.chr, call.breakpointB.pos
        startA, endA = call.breakpointA.start, call.breakpointA.end
        startB, endB = call.breakpointB.start, call.breakpointB.end
        category = call.category.value
        resolution = call.resolution
        neg_log_pval = call.neg_log_pval
    else:
        chrA, posA = call.chrA, (call.startA + call.endA) // 2
        chrB, posB = call.chrB, (call.startB + call.endB) // 2
        startA, endA = call.startA, call.endA
        startB, endB = call.startB, call.endB
        category = "N/A"
        resolution = "N/A"
        neg_log_pval = "N/A"

    # Get genes at call
    genesA, genesB = get_genes_at_call(call)

    # Get nearby genes
    nearbyA, nearbyB = get_genes_around_call(call)

    # Get breakfinder submatrix as string
    submatrix_string = f"{chrA}:{startA}-{endA} x {chrB}:{startB}-{endB}"

    # Geneerate direct and nearby genes
    genesetA = set(genesA)
    genesetB = set(genesB)
    html_genesA = "\n".join([make_html_gene(gene, direct=True) for gene in genesA])
    html_genesA += "\n".join(
        [make_html_gene(gene, direct=False) for gene in nearbyA if gene not in genesetA]
    )

    html_genesB = "\n".join([make_html_gene(gene, direct=True) for gene in genesB])
    html_genesB += "\n".join(
        [make_html_gene(gene, direct=False) for gene in nearbyB if gene not in genesetB]
    )

    # Generate call_region plot
    calL_region_plot_fig = plot_composite_context_and_zoom(
        sample, call, extra_bedpe=extra_bedpe_data, extra_bigwig_handles=extra_bigwig_handles, crosshairs=crosshairs, grid=grid, plot_at_call_resolution=True
    )
    call_region_plot_base64 = fig_to_base64_and_close(calL_region_plot_fig)

    multi_res_plot_fig = plot_composite_multires_breakpoint(
        sample, call, extra_bedpe=extra_bedpe_data,
    )
    multi_res_plot_base64 = fig_to_base64_and_close(multi_res_plot_fig)

    # Maybe generate control comparison plot
    html_maybe_comparison = make_html_comparison_plot(sample, control, call)

    # Replace template
    return TEMPLATE_CALL.format(
        call_region_string=call_to_string(call),
        call_region_string_no_spaces=call_to_alphanumeric_string(call),
        category=category,
        submatrix_string=submatrix_string,
        call_resolution=resolution,
        neg_log_p=neg_log_pval,
        html_genesA=html_genesA,
        html_genesB=html_genesB,
        chrA=chrA,
        chrB=chrB,
        call_region_plot_base64=call_region_plot_base64,
        multi_res_plot_base64=multi_res_plot_base64,
        html_maybe_comparison=html_maybe_comparison,
    )


def make_html_report(
    sample_id: str,
    hic_filepath: str,
    qc_filepath: str | None = None,
    breakfinder_filepath: str | None = None,
    control_filepath: str | None = None,
    extra_bedpe: list[str] = [],
    extra_bigwig: list[str] = [],
    crosshairs: bool = False,
    grid: bool=False,
) -> str:
    """Generate HTML for a full report"""

    plt.ioff()

    sample = read_sample(sample_id, hic_filepath, qc_filepath, breakfinder_filepath)
    extra_bedpe_data = [read_bedpe(filepath) for filepath in extra_bedpe]
    extra_bigwig_handles = [(os.path.basename(filepath), pyBigWig.open(filepath)) for filepath in extra_bigwig]

    print(
        f"Hi-C sample loaded: {sample.id}. Generating report (this may take a while)..."
    )

    if sample.breakfinder_calls is not None:
        print(
            f"There are {len(sample.breakfinder_calls)} breakfinder calls in this sample."
        )

    if control_filepath is not None:
        control = read_sample("", control_filepath, None, None)
    else:
        control = None

    if len(extra_bedpe) == 0:
        bedpe_filepaths = "None"
    else:
        bedpe_filepaths = ""
        for filepath, color in zip(extra_bedpe, BEDPE_COLORS):
            bedpe_filepaths += f"<div style='background: {color}; height: 1rem; width: 1rem; display: inline-block; margin-right: 1rem;'></div>{filepath}<br>"

    if len(extra_bigwig) == 0:
        bigwig_filepaths = "None"
    else:
        bigwig_filepaths = ""
        for filepath in extra_bigwig:
            bigwig_filepaths += f"{filepath}<br>"

    if sample.breakfinder_calls is not None:
        # Generate call entries
        html_calls = "\n".join(
            [
                make_html_call(sample, call, control, extra_bedpe_data, extra_bigwig_handles, crosshairs, grid=grid)
                for call in sample.breakfinder_calls
            ]
        )
        # Generate sidebar call entries
        html_sidebar_calls = "\n".join(
            [make_html_sidebar_call(call) for call in sample.breakfinder_calls]
        )
    else:
        html_calls = "No breakfinder calls were provided."
        html_sidebar_calls = ""

    # Generate qc plot
    html_maybe_qc_plot = make_html_qc_plot(sample)

    # Generate full Hi-C matrix
    hic_matrix_fig = plot_composite_double_whole_matrix(sample)
    full_hic_matrix_base64 = fig_to_base64_and_close(hic_matrix_fig)

    print("Report generation complete.")

    # Replace template
    return TEMPLATE_REPORT.format(
        sample_id=sample.id,
        hic_filepath=hic_filepath,
        qc_filepath=qc_filepath,
        breakfinder_filepath=breakfinder_filepath,
        bedpe_filepaths=bedpe_filepaths,
        bigwig_filepaths=bigwig_filepaths,
        control_hic_filepath=control_filepath,
        generation_datetime=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        html_sidebar_sub_list=html_sidebar_calls,
        html_maybe_qc_plot=html_maybe_qc_plot,
        full_hic_matrix_base64=full_hic_matrix_base64,
        html_breakfinder_calls=html_calls,
    )


def make_html_report_and_save(
    sample_id: str,
    hic_filepath: str,
    qc_filepath: str | None = None,
    breakfinder_filepath: str | None = None,
    extra_bedpe: str | None = None,
    extra_bigwig: str | None = None,
    crosshairs: bool=False,
    grid: bool=False,
    control_filepath: str | None = None,
    output_filepath: str | None = None,
) -> None:
    """Generate a full report and save it to a file"""

    if output_filepath is None:
        output_filepath = f"{sample_id}_report.html"

    if not output_filepath.endswith(".html"):
        output_filepath += ".html"

    if extra_bedpe is not None:
        # Split by commas
        extra_bedpe = extra_bedpe.split(",")
    else:
        extra_bedpe = []

    if extra_bigwig is not None:
        # Split by commas
        extra_bigwig = extra_bigwig.split(",")
    else:
        extra_bigwig = []

    if not os.path.exists(hic_filepath):
        raise FileNotFoundError(f"Hi-C file not found at {hic_filepath}.")

    if control_filepath is not None and not os.path.exists(control_filepath):
        raise FileNotFoundError(f"Control Hi-C file not found at {control_filepath}.")

    report_html = make_html_report(
        sample_id,
        hic_filepath,
        qc_filepath,
        breakfinder_filepath,
        control_filepath,
        extra_bedpe=extra_bedpe,
        extra_bigwig=extra_bigwig,
        crosshairs=crosshairs,
        grid=grid,
    )

    with open(output_filepath, "w") as f:
        f.write(report_html)

    print(f"Report saved to {output_filepath}.")
