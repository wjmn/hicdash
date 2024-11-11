"""Functions for generating Hi-C dashboard / report. 
"""

from pathlib import Path
from jinja2 import Template
import matplotlib.pyplot as plt
import base64
import io
import datetime
import os
import pyBigWig # type: ignore

from hicdash.constants import *
from hicdash.utilities import *
from hicdash.definitions import *
from hicdash.plotters import *


# Load templates
with open(Path(__file__).parent / "templates/result.jinja") as f:
    jinja_template_result = Template(f.read())

with open(Path(__file__).parent / "templates/report.jinja") as f:
    jinja_template_full = Template(f.read())


def fig_to_base64_and_close(fig: plt.Figure, dpi=72) -> str:
    """Converts a matplotlib figure to a base64 string"""

    # Save figure to bytes
    fig_io_bytes = io.BytesIO()
    fig.savefig(fig_io_bytes, format="png", bbox_inches="tight", dpi=dpi)
    fig_io_bytes.seek(0)
    fig_hash = base64.b64encode(fig_io_bytes.read())

    plt.close(fig)

    return fig_hash.decode("utf-8")

def make_gene_dicts(bpoint: Breakpoint, radius=1000000, keygenes=[], is_protein_coding_only=True, allow_ig=True,) -> tuple[list[dict], list[dict]]:
    """Create a list of dictionaries for genesA and genesB for jinja template.

    Each dictionary contains:
    - name: gene name or ID
    - is_involved: whether the gene is involved in the breakpoint
    - is_key: whether the gene is a key gene (in provided keygenes list)
    """
    
    regionA, regionB = bpoint.get_reconstructed_regions_with_radius(radius=radius)
    genesA = sorted(regionA.get_contained_genes(), key=lambda x: 0 if (x.start < bpoint.breakendA.pos < x.end) else min(abs(x.start - bpoint.breakendA.pos), abs(x.end - bpoint.breakendA.pos)))
    genesB = sorted(regionB.get_contained_genes(), key=lambda x: 0 if (x.start < bpoint.breakendB.pos < x.end) else min(abs(x.start - bpoint.breakendB.pos), abs(x.end - bpoint.breakendB.pos)))

    intersectA = set(s.gene_name for s in bpoint.breakendA.get_intersecting_genes())
    intersectB = set(s.gene_name for s in bpoint.breakendB.get_intersecting_genes())

    keygenes_set = set(keygenes)

    dictsA = []
    dictsB = []

    for dict_list, gene_list, intersect_list in zip([dictsA, dictsB], [genesA, genesB], [intersectA, intersectB]):
        seen_ig = set()
        for gene in gene_list:
            is_involved = gene.gene_name in intersect_list
            is_key = gene.gene_name in keygenes_set
            name = get_gene_name_or_id(gene)
            if is_protein_coding_only:
                if gene.biotype != "protein_coding":
                    if allow_ig and is_ig_gene(gene):
                        ig_region = gene.gene_name[:3]
                        if ig_region in seen_ig or ig_region == "":
                            continue
                        else:
                            name = ig_region + " Region"
                            seen_ig.add(ig_region)
                            is_key = True
                    else:
                        continue
            dict_list.append({
                "name": name,
                "is_involved": is_involved,
                "is_key": is_key,
            })
    return (dictsA, dictsB)

def make_result_dict_from_bpoint(sample, bpoint: Breakpoint, geneA=None, geneB=None, keyNearA=None, keyNearB=None, keygenes=[], gene_radius=1000000, control_sample=None) -> dict:
    """Create a dictionary for jinja template from a single breakpoint.
    """
    
    fig_zoom = plot_composite_context_and_zoom(sample, bpoint, crosshairs=True, plot_at_bpoint_resolution=True, keygenes=keygenes, capped_resolution=10000)
    fig_multires = plot_composite_multires_breakpoint(sample, bpoint)
    fig_triangle = plot_composite_triangle(sample, bpoint, keygenes=keygenes)

    if control_sample is None:
        fig_comparison = None
    else:
        fig_comparison = plot_composite_compare_two(sample, control_sample, bpoint)

    nearbyA, nearbyB = make_gene_dicts(bpoint, radius=gene_radius, keygenes=keygenes)

    template_dict = {
        "regionA": str(bpoint.breakendA),
        "regionB": str(bpoint.breakendB),
        "chromA": bpoint.breakendA.chrom,
        "chromB": bpoint.breakendB.chrom,
        "nearbyGenesA": nearbyA, 
        "nearbyGenesB": nearbyB,
        "base64_figure_triangle": fig_to_base64_and_close(fig_triangle),
        "base64_figure_zoom": fig_to_base64_and_close(fig_zoom),
        "base64_figure_multires": fig_to_base64_and_close(fig_multires),
    }

    if fig_comparison is not None:
        template_dict["base64_figure_comparison"] = fig_to_base64_and_close(fig_comparison)

    if geneA is not None and geneB is not None:
        template_dict["geneA"] = geneA
        template_dict["geneB"] = geneB
    elif keyNearA is not None and keyNearB is not None:
        template_dict["keyNearA"] = keyNearA
        template_dict["keyNearB"] = keyNearB

    return template_dict


def render_sample_report(
    prefix=None,
    hic_filepath=None,
    qc_filepath=None,
    breakpoints_filepath=None,
    extra_bedpe=None,
    extra_bigwig=None,
    control_filepath=None,
    key_genes_filepath=None,
    targets1_filepath=None,
    targets2_filepath=None,
    output_filepath=None,
    command=None,
    flagged_filepath=None,
):
    """Render a full report for a single sample run and save to output_filepath.
    
    Extra files:
    - key_genes_filepath should point to a file containing a line-separated list of key genes (one gene name per line).
    - targets1_filepath should point to a file containing a line-separated list of genes for single-target pseudotargeted Hi-C. 
    - targets2_filepath should point to a file containing a line-separated list of gene pairs (separated by tabs) for dual-target pseudotargeted Hi-C.

    """

    plt.ioff()

    sample = ArimaPipelineSample(prefix, hic_filepath, qc_filepath, breakpoints_filepath)

    if control_filepath is not None:
        control_id = control_filepath.split("/")[-1].split(".")[0].strip("_inter_30")
        control = ArimaPipelineSample(control_id, control_filepath, None, None) if control_filepath is not None else None
    else:
        control = None

    if flagged_filepath is not None:
        flagged = PairedRegion.list_from_bedpe_file(flagged_filepath, skiprows=0)
    else:
        flagged = []

    now = datetime.datetime.strftime( datetime.datetime.now(), "%Y-%m-%d %H:%M:%S")
    print(f"Generating report for {sample.id} at {now}.")


    # Read gene list files
    if key_genes_filepath is None:
        keygenes = []
    else:
        with open(key_genes_filepath) as f:
            keygenes = [line.strip() for line in f.readlines()]

    if targets1_filepath is None:
        single_targets = []
    else:
        with open(targets1_filepath) as f:
            single_targets = [line.strip() for line in f.readlines()]

    if targets2_filepath is None:
        dual_targets = []
    else: 
        with open(targets2_filepath) as f:
            dual_targets = [line.strip().split("\t") for line in f.readlines()]

    # Make figures & extract bas64 strings        

    # QC plot
    fig_qc = plot_qc(sample)
    base64_figure_qc = fig_to_base64_and_close(fig_qc, dpi=96)

    # Full Hi-C matrix
    fig_hic_whole, ax_hic_whole = plt.subplots(figsize=(9,9))
    _ = plot_full_matrix(sample, show_breakpoints=True, ax=ax_hic_whole)
    base64_figure_full_hic = fig_to_base64_and_close(fig_hic_whole, dpi=96)
    
    # Results, split into categories
    gene_fusions = []
    near_keygenes = []
    other_bpoints = []
    flagged_bpoints = []

    print(f"Processing {len(sample.breakpoints)} breakpoints. This may take a while.")
    
    for bpoint in sorted(sample.breakpoints):
        # Check if in flagged
        is_flagged = False
        for flagged_region in flagged:
            if (bpoint.breakendA.is_inside(flagged_region.regionA) and bpoint.breakendB.is_inside(flagged_region.regionB)) or \
                (bpoint.breakendA.is_inside(flagged_region.regionB) and bpoint.breakendB.is_inside(flagged_region.regionA)):
                is_flagged = True
                break
        if is_flagged:
            flagged_bpoints.append(bpoint)
            continue
        # Check if gene fusion
        fusions = bpoint.get_possible_gene_fusions()
        if len(fusions) > 0:
            for fusion in fusions:
                gene_fusions.append((fusion, bpoint))
        else:
            keyNearA, keyNearB = bpoint.get_nearby_key_genes(radius=1000000, key_genes=keygenes)
            if len(keyNearA) > 0 or len(keyNearB) > 0:
                near_keygenes.append(((list(keyNearA), list(keyNearB)), bpoint))
            else:
                other_bpoints.append(bpoint)

    rendered_gene_fusion = []
    rendered_keygenes = []
    rendered_additional = []
    rendered_flagged = []
    processed_count = 0 
    for ((geneA, geneB), bpoint) in gene_fusions:
        result_dict = make_result_dict_from_bpoint(sample, bpoint, keygenes=keygenes, geneA=get_gene_name_or_id(geneA), geneB=get_gene_name_or_id(geneB), control_sample=control)
        rendered_gene_fusion.append(jinja_template_result.render(result_dict))

        processed_count += 1
        print(f"Processed {processed_count}/{len(gene_fusions)} possible gene fusions.")

    processed_count = 0 
    for ((keyNearA, keyNearB), bpoint) in near_keygenes:
        result_dict = make_result_dict_from_bpoint(sample, bpoint, keygenes=keygenes, keyNearA=", ".join(keyNearA), keyNearB=", ".join(keyNearB), control_sample=control)
        rendered_keygenes.append(jinja_template_result.render(result_dict))

        processed_count += 1
        print(f"Processed {processed_count}/{len(near_keygenes)} results near key genes.")

    processed_count = 0 
    for bpoint in other_bpoints:
        result_dict = make_result_dict_from_bpoint(sample, bpoint, control_sample=control)
        rendered_additional.append(jinja_template_result.render(result_dict))

        processed_count += 1
        print(f"Processed {processed_count}/{len(other_bpoints)} additional findings.")

    processed_count = 0 
    for bpoint in flagged_bpoints:
        result_dict = make_result_dict_from_bpoint(sample, bpoint, control_sample=control)
        rendered_flagged.append(jinja_template_result.render(result_dict))

        processed_count += 1
        print(f"Processed {processed_count}/{len(flagged_bpoints)} flagged false positives.")

    print(f"Generating figures for {len(single_targets)} single-target pseudotargets and {len(dual_targets)} dual-target pseudotargets.")

    base64_list_single_target = []
    base64_list_dual_target = []
    for target in single_targets:
        fig, positive = plot_pseudotarget_hic_single(sample, target)
        base64_list_single_target.append(fig_to_base64_and_close(fig))
    for (geneA, geneB) in dual_targets:
        fig = plot_pseudotarget_hic_dual(sample, geneA, geneB)
        base64_list_dual_target.append(fig_to_base64_and_close(fig))

    print("All figures generated. Generating report and saving to file.")

    d = {
        "sample_id": sample.id,
        "generation_datetime": now,
        "prefix": prefix,
        "hic_filepath": hic_filepath,
        "qc_filepath": qc_filepath,
        "breakpoints_filepath": breakpoints_filepath,
        "extra_bedpe": extra_bedpe,
        "extra_bigwig": extra_bigwig,
        "control_filepath": control_filepath,
        "key_genes": key_genes_filepath,
        "targets1": targets1_filepath,
        "targets2": targets2_filepath,
        "output_filepath": output_filepath,
        "command": command,
        "base64_figure_qc": base64_figure_qc,
        "base64_figure_full_hic": base64_figure_full_hic,
        "rendered_results_gene_fusion": rendered_gene_fusion,
        "rendered_results_near_keygenes": rendered_keygenes,
        "rendered_results_additional": rendered_additional,
        "rendered_results_flagged": rendered_flagged,
        "base64_list_single_target": base64_list_single_target,
        "base64_list_dual_target": base64_list_dual_target,
    }

    rendered = jinja_template_full.render(d)

    with open(output_filepath, "w") as f:
        f.write(rendered)


    print(f"Report saved to {output_filepath}.")