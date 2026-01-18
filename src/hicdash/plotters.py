"""Plotting functions used for dashboard generation. 

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from numpy.typing import NDArray
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar # type: ignore
from hicstraw import HiCFile # type: ignore
from hicdash.constants import CHROM_SIZES, CHROMS, CHROM_INDICES, GENE_ANNOTATIONS, MARKER_SIZE_DICT, CHROM_COLORS, BIGWIG_COLORS
from hicdash.definitions import ArimaPipelineSample, Breakpoint, GenomicRegion, Strand, AssembledHic, PlotRegion
from matplotlib.patches import Rectangle, Ellipse
from matplotlib.ticker import FixedLocator, MultipleLocator
from hicdash.utilities import *
import pandas as pd
import matplotlib


# Default red colormap for all Hi-C plots (similar to Juicebox) with gray masking
REDMAP = LinearSegmentedColormap.from_list("bright_red", [(1, 1, 1), (1, 0, 0)])
REDMAP.set_bad(color="gainsboro")


def plot_hic_region_matrix(
    sample: ArimaPipelineSample,
    regionX: GenomicRegion,
    regionY: GenomicRegion,
    resolution: int,
    measure="observed",
    normalization="NONE",
    norm_constant_normalize=False,
    ax: plt.Axes | None=None,
    aspect="auto",
    minimal=False,
    show_breakpoints=True,
    breakpoint_highlight:Breakpoint | None=None, 
    vmax=None,
    cmap=REDMAP,
    title="",
    title_fontsize=11,
    label_fontsize=10,
    tick_fontsize=9,
    grid=False,
    crosshairs=False,
    show_submatrices=False,
    mew=None,
    marker_size=None,
) -> None:
    """Plots a specified Hi-C region.

        plot_hic_region_matrix(sample, regionX, regionY, resolution, ax=ax)

    regionX and regionY must be bin-aligned to a multiple of resolution). 

    If vmax is None, it will be sent to a sane default:
    - Max value in the data matrix if the regions are inter-chromosomal
    - Max value in the data matrix if the regions are intra-chromosomal but don't include the diagonal
    - Sqrt of the max value in the data matrix if the region includes main diagonal

    The "origin" is at the top left corner (i.e. minX and minY are at the top left corner). 

    """
    
    # Assert that regions must be resolution-aligned
    assert regionX.bin_align_res is not None and regionX.bin_align_res % resolution == 0
    assert regionY.bin_align_res is not None and regionY.bin_align_res % resolution == 0

    # Assert that regions must not be reversed
    assert not regionX.reverse
    assert not regionY.reverse

    # Get plot axis (or get global axis if none provided)
    if ax is None:
        ax = plt.gca()

    # Get masked matrix data
    data = sample.get_hic_region_data(regionX, regionY, resolution, measure=measure, normalization=normalization, norm_constant_normalize=norm_constant_normalize)

    # Decide on vmax if not already set
    if vmax is None:
        # Set color scale max depending if interchromosomal or intrachromosomal and on resolution
        if regionX.chrom == regionY.chrom:
            # If intra-chromosomal, check whether the diagonal is included in the plot
            if regionX.overlaps(regionY):
                # If diagonal is present in the plot, then set vmax to sqrt of the vax
                vmax = np.sqrt(data.max())
            else:
                # If diagonal is not present in the plot, then relax vmax
                vmax = data.max()
        else:
            # If inter-chromosomal, then set max as simply vmax
            vmax = data.max()
    if vmax == 0:
        vmax = 1

    # Unpack region for convenience
    chromX, startX, endX = regionX.get_unpacked()
    chromY, startY, endY = regionY.get_unpacked()
    centerX = (startX + endX) // 2
    centerY = (startY + endY) // 2

    # Plot the matrix
    plot_startX, plot_endX = (startX, endX) if not regionX.reverse else (endX, startX)
    plot_startY, plot_endY = (startY, endY) if not regionY.reverse else (endY, startY)
    im = ax.matshow(data, cmap=cmap, vmin=0, vmax=vmax, aspect=aspect, extent=[plot_startX, plot_endX, plot_endY, plot_startY])
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Add axis labels (depending on level of detail)
    if minimal:
        # Minimal plot just has the hic matrix data and chromosome labels
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontdict={"fontsize": title_fontsize})
        ax.set_xlabel(f"{chromX}", fontdict={"fontsize": label_fontsize})
        ax.set_ylabel(f"{chromY}", fontdict={"fontsize": label_fontsize})

    else:
        # Make a start, center and end tick for each axis (and align appropriately on the axis)
        ax.set_xticks([startX, centerX, endX], map(lambda x: f"{x:,}", [startX, centerX, endX]))
        ax.set_yticks([startY, centerY, endY], map(lambda x: f"{x:,}", [startY, centerY, endY]), rotation=90)

        # Label the axes
        ax.set_xlabel(f"{chromX}", fontdict={"fontsize": label_fontsize})
        ax.set_ylabel(f"{chromY}", fontdict={"fontsize": label_fontsize}, rotation=90)
        ax.tick_params(axis="both", which="major", labelsize=tick_fontsize)

        # Set tick param customizations
        ax.xaxis.set_tick_params(which="major", length=5)
        ax.yaxis.set_tick_params(which="major", length=5)
        ax.xaxis.tick_bottom()
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")

        # Align the tick labels neatly
        xticklabels = ax.get_xticklabels()
        xticklabels[0].set_horizontalalignment("left")
        xticklabels[0].set_fontsize(tick_fontsize - 1)
        xticklabels[-1].set_horizontalalignment("right")
        xticklabels[-1].set_fontsize(tick_fontsize - 1)

        yticklabels = ax.get_yticklabels()
        yticklabels[0].set_verticalalignment("top")
        yticklabels[0].set_fontsize(tick_fontsize - 1)
        yticklabels[1].set_verticalalignment("center")
        yticklabels[-1].set_verticalalignment("bottom")
        yticklabels[-1].set_fontsize(tick_fontsize - 1)

        # Annotate the resolution of the heatmap
        scalebar = AnchoredSizeBar(
            ax.transData,
            resolution,
            int_to_resolution(resolution),
            "lower left",
            pad=0.5,
            frameon=False,
            fontproperties={"size": tick_fontsize},
            size_vertical=resolution,
        )
        ax.add_artist(scalebar)

    # Plot grid lines if specified (mostly just for tests)
    if grid:
        xminor_ticks = FixedLocator(np.arange(startX, endX, resolution))
        yminor_ticks = FixedLocator(np.arange(startY, endY, resolution))
        ax.xaxis.set_minor_locator(xminor_ticks)
        ax.yaxis.set_minor_locator(yminor_ticks)
        ax.grid(True, which="both", linestyle="solid", linewidth=0.5, color="gainsboro")

    # Plot annotations
    annotation_color = "black"
    if show_breakpoints and sample.breakpoints is not None:
        for bpoint in sample.breakpoints:
            if bpoint.breakendA.chrom == chromX and bpoint.breakendB.chrom == chromY:
                posX = bpoint.breakendA.pos
                strandX = bpoint.breakendA.strand
                bpoint_regionX = bpoint.break_regionA
                posY = bpoint.breakendB.pos
                strandY = bpoint.breakendB.strand
                bpoint_regionY = bpoint.break_regionB
            elif bpoint.breakendB.chrom == chromX and bpoint.breakendA.chrom == chromY:
                posX = bpoint.breakendB.pos
                strandX = bpoint.breakendB.strand
                bpoint_regionX = bpoint.break_regionB
                posY = bpoint.breakendA.pos
                strandY = bpoint.breakendA.strand
                bpoint_regionY = bpoint.break_regionA
            else:
                continue
                
            alpha = 1
            size = MARKER_SIZE_DICT[resolution] if marker_size is None else marker_size
            mew=size / 5 if mew is None else mew
            if breakpoint_highlight is not None and bpoint != breakpoint_highlight and mew is None:
                alpha = 0.6
                mew *= 0.6
                # size *= 0.6

            # If the breakpoint is within the bounds of the plot, plot it
            if startX <= posX <= endX and startY <= posY <= endY:
                if show_submatrices and bpoint_regionX is not None and bpoint_regionY is not None:
                    rect = Rectangle((bpoint_regionX.start, bpoint_regionY.start), bpoint_regionX.get_size(), bpoint_regionY.get_size(), linewidth=0.5, edgecolor=annotation_color, facecolor="none", alpha=alpha,)
                    ax.add_patch(rect)
                # Match marker based on strandness
                match strandX:
                    case Strand.POS:
                        markerX = 0 #tickleft
                    case Strand.NEG:
                        markerX = 1 #tickright
                match strandY:
                    case Strand.POS:
                        markerY = 2 #tickup
                    case Strand.NEG:
                        markerY = 3 #tickdown
                ax.plot(posX, posY, marker=markerX, color=annotation_color, markersize=size, mew=mew, alpha=alpha,)
                ax.plot(posX, posY, marker=markerY, color=annotation_color, markersize=size, mew=mew, alpha=alpha,)
                if crosshairs:
                    ax.axvline(posX, color=annotation_color, linestyle=(0, (1, 3)), linewidth=mew, alpha=0.5)
                    ax.axhline(posY, color=annotation_color, linestyle=(0, (1, 3)), linewidth=mew, alpha=0.5)

    # Add inset colorbar
    cax = ax.inset_axes((0.05, 0.90, 0.1, 0.05))
    vmax_label = str(int(round(vmax))) if (isinstance(vmax, int) or abs(vmax-round(vmax)) < 0.001) else f"{vmax:.2f}" if vmax < 1 else f"{vmax:.1f}"
    balancing = SHORT_NORM[normalization]
    plt.colorbar(im, cax=cax, orientation="horizontal", )
    cax.set_xticks([])
    cax.set_title(f"{vmax_label} ({balancing})", x=1.3, y=0.42, ha="left", va="center", transform=cax.transAxes, fontsize=tick_fontsize-1)

    # Reset x and y lim, in case the plotting of the markers changed it
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


def plot_hic_chr_context(
    sample: ArimaPipelineSample,
    chromA: str,
    chromB: str,
    resolution: int = 2500000,
    show_breakpoints=True,
    region_highlight: tuple[GenomicRegion, GenomicRegion] | None = None,
    measure="observed",
    normalization="NONE",
    norm_constant_normalize=False,
    ax=None,
    cmap=REDMAP,
    tick_fontsize=10,
    vmax=None,
    aspect="auto",
) -> None:
    """Plots the Hi-C whole-chromosome context for a given sample.

        plot_hic_chr_context(sample, chromA, chromB, ax=ax)

    If chromA and chromB are the same, the plot will be a single chromosome plot.
    Otherwise, it will be a 4-box plot showing both intra-chromosomal and inter-chromosomal whole-chromosome views.

    Highlights and bpoints are shown on the bottom left side of the diagonal; the top right side of the diagonal is unannotated.

    if vmax is None, then defaults to square root of the max data value. 

    """

    # Get axes
    if ax is None:
        ax = plt.gca()

    # Get HiC Data for each box
    regionA = GenomicRegion.from_chrom(chromA)
    regionB = GenomicRegion.from_chrom(chromB)

    # Retrieve each box of data
    top_left = sample.get_hic_direct_data(regionA, regionA, resolution, measure=measure, normalization=normalization, norm_constant_normalize=norm_constant_normalize)
    bottom_left = sample.get_hic_direct_data(regionA, regionB, resolution, measure=measure, normalization=normalization, norm_constant_normalize=norm_constant_normalize)
    top_right = bottom_left.T
    bottom_right = sample.get_hic_direct_data(regionB, regionB, resolution, measure=measure, normalization=normalization, norm_constant_normalize=norm_constant_normalize)

    # Stack the Hi-C data into one 4-box matrix
    top_row = np.hstack([top_left, top_right])
    bottom_row = np.hstack([bottom_left, bottom_right])
    full_matrix = np.vstack([top_row, bottom_row])

    # Calculate max colormap value
    if vmax is None:
        max_value = np.max(full_matrix)
        vmax_large = np.sqrt(max_value)
    else:
        vmax_large = vmax

    # Plot the large matrix
    im = ax.matshow(full_matrix, cmap=cmap, vmax=vmax_large, aspect=aspect)

    # Add axis lines to separate the chromosomes
    ax.axhline(top_left.shape[0] - 0.5, color="gray", linewidth=1)
    ax.axvline(top_left.shape[1] - 0.5, color="gray", linewidth=1)

    # Add chromosome tick labels
    ticks = [top_left.shape[1] // 2, top_left.shape[1] + top_right.shape[1] // 2]
    ax.set_xticks(ticks, [chromA, chromB], fontsize=tick_fontsize)
    ax.set_yticks(ticks, [chromA, chromB], fontsize=tick_fontsize, rotation=90, va="center")
    ax.xaxis.set_ticks_position("bottom")

    if show_breakpoints and sample.breakpoints is not None:

        box_width = 0.01 * full_matrix.shape[0] * 2
        box_height = 0.01 * full_matrix.shape[1] * 2
        if chromA == chromB:
            box_width /= 2
            box_height /= 2

        # Choose only bpoints that involve these two chromosomes
        for bpoint in sample.breakpoints:

            bpoint_chromA = bpoint.breakendA.chrom
            bpoint_chromB = bpoint.breakendB.chrom
            bpoint_posA = bpoint.breakendA.pos
            bpoint_posB = bpoint.breakendB.pos
            if bpoint_chromA == chromA and bpoint_chromB == chromB:
                # Get position of the breakfinder bpoint as a fraction of the chroomsome, then multiply by matrix size and offset by 0.5
                box_top = (top_left.shape[0] + ((bpoint_posB / CHROM_SIZES[chromB]) * bottom_left.shape[0]) - 0.5 )
                box_left = bpoint_posA / CHROM_SIZES[chromA] * top_left.shape[1] - 0.5
            elif bpoint_chromA == chromB and bpoint_chromB == chromA:
                box_top = ( top_left.shape[0] + ((bpoint_chromA / CHROM_SIZES[chromA]) * bottom_left.shape[0]) - 0.5 )
                box_left = bpoint_chromB / CHROM_SIZES[chromB] * top_left.shape[1] - 0.5
            elif chromA != chromB and bpoint_chromA == chromA and bpoint_chromB == chromA:
                box_top = (((bpoint_posB / CHROM_SIZES[chromA]) * top_left.shape[0]) - 0.5 )
                box_left = bpoint_posA / CHROM_SIZES[chromA] * top_left.shape[1] - 0.5
            elif chromA != chromB and bpoint_chromA == chromB and bpoint_chromB == chromB:
                box_top = (top_left.shape[0] + ((bpoint_posB / CHROM_SIZES[chromB]) * bottom_left.shape[0]) - 0.5 )
                box_left = (bottom_left.shape[1] + (bpoint_posA / CHROM_SIZES[chromB] * bottom_right.shape[1] - 0.5))
            else:
                continue

            # Add the breakfinder bpoint to the plot on the bottom left side of the diagonal
            ax.add_patch( Ellipse( (box_left, box_top), box_width, box_height, fill=False, edgecolor="black", linewidth=0.5, ) )

    if region_highlight is not None:

        box_width = 0.01 * full_matrix.shape[0] * 2
        box_height = 0.01 * full_matrix.shape[1] * 2
        if chromA == chromB:
            box_width /= 2
            box_height /= 2

        regionX, regionY = region_highlight
        startX, endX = regionX.start, regionX.end
        startY, endY = regionY.start, regionY.end
        posA = (startX + endX) // 2
        posB = (startY + endY) // 2

        # Plot in the bottom left box
        box_top = ( top_left.shape[0] + ((posB / CHROM_SIZES[chromB]) * bottom_left.shape[0]) - 0.5 )
        box_left = posA / CHROM_SIZES[chromA] * top_left.shape[1] - 0.5
        ax.add_patch( Ellipse( (box_left, box_top), box_width, box_height, fill=False, edgecolor="blue", linewidth=1, ) )

    # If chromA == chromB, then show only the bottom left box (the four boxes will essentially just be all the same box, so you can just halve the axis limits)
    if chromA == chromB:
        xmin, xmax = ax.get_xlim()
        ax.set_xlim(-0.5, (xmax + 0.5) // 2 - 0.5)
        ax.set_ylim((xmax + 0.5) // 2 - 0.5, xmax)
        ax.invert_yaxis()

    # Annotate the resolution of the heatmap
    bin_size = resolution * full_matrix.shape[0] / (CHROM_SIZES[chromA] + CHROM_SIZES[chromB])
    # print(bin_size)
    scalebar = AnchoredSizeBar(
        ax.transData,
        bin_size,
        int_to_resolution(resolution),
        "upper right",
        pad=0.5,
        frameon=False,
        fontproperties={"size": tick_fontsize},
        size_vertical=bin_size,
    )
    ax.add_artist(scalebar) 

    cax = ax.inset_axes([0.70, 0.95, 0.05, 0.02])
    vmax_label = str(int(round(vmax_large))) if (isinstance(vmax_large, int) or abs(vmax_large-round(vmax_large)) < 0.001) else f"{vmax_large:.2f}" if vmax_large < 1 else f"{vmax_large:.1f}"
    plt.colorbar(im, cax=cax, orientation="horizontal", )
    cax.set_xticks([])
    cax.set_title(f"{vmax_label}", x=1.3, y=0.42, ha="left", va="center", transform=cax.transAxes, fontsize=tick_fontsize)


def plot_full_matrix(
    sample: ArimaPipelineSample,
    ax=None,
    show_breakpoints=False,
    cmap=REDMAP,
    vmax=None,
) -> None:
    """Plot the full Hi-C matrix, with or without annotations on breakfinder bpoints.

        plot_full_matrix(sample, ax=ax)

    if vmax is None, defaults to square root of the maximum data value. 
    
    """

    # Collate all hic data at 2500000 resolution
    rows = []
    for chrY in CHROMS:
        row = []
        for chrX in CHROMS:
            data = sample.get_hic_direct_data(
                GenomicRegion.from_chrom(chrX),
                GenomicRegion.from_chrom(chrY),
                2500000
            )
            row.append(data)
        rows.append(row)

    # Get size of all chromosome matrices
    new_sizes = [c.shape[1] for c in rows[0]]

    # Combine into a single matrix
    full_matrix = np.vstack([np.hstack(row) for row in rows])

    # Get axis
    if ax is None:
        ax = plt.gca()

    # Apply colour scale (fraction of sqrt of max value for now)
    if vmax is None:
        max_value = np.max(full_matrix)
        vmax = np.sqrt(max_value)

    # Plot matrix
    im = ax.matshow(full_matrix, cmap=cmap, vmax=vmax)

    # Add chromosome ticks at center of each chromosome matrix
    tick_positions = []
    cumsum = -0.5
    for i in range(len(CHROMS)):
        ax.axvline(cumsum + new_sizes[i], color="gray", linewidth=0.5)
        ax.axhline(cumsum + new_sizes[i], color="gray", linewidth=0.5)
        tick_positions.append(cumsum + new_sizes[i] / 2)
        cumsum += new_sizes[i]
    ax.set_xticks(tick_positions, CHROMS, rotation=90, fontsize=6)
    ax.set_yticks(tick_positions, CHROMS, fontsize=6)
    ax.xaxis.set_ticks_position("top")

    if show_breakpoints and sample.breakpoints is not None:
        for bpoint in sample.breakpoints:
            chrY = bpoint.breakendA.chrom
            chrX = bpoint.breakendB.chrom
            posA = bpoint.breakendA.pos
            posB = bpoint.breakendB.pos

            # Show only inter-chromosomal bpoints for now
            # if bpoint.is_inter():

            # Calculate position on full-matrix as a fraction of the chromosome
            # Show only on lower left triangle
            idxA = CHROM_INDICES[chrY]
            pctA = posA / CHROM_SIZES[chrY]
            coordA = sum(new_sizes[:idxA]) + (pctA * new_sizes[idxA])

            idxB = CHROM_INDICES[chrX]
            pctB = posB / CHROM_SIZES[chrX]
            coordB = sum(new_sizes[:idxB]) + (pctB * new_sizes[idxB])

            radius = 0.02 * full_matrix.shape[1]
            ellipse2 = Ellipse(
                (coordA, coordB),
                radius,
                radius,
                fill=False,
                edgecolor="blue",
                linewidth=1,
            )
            ax.add_patch(ellipse2)



def plot_gene_track(
    region: GenomicRegion,
    gene_filter: list[str] | None = None,
    hide_axes=True,
    vertical=False,
    ax=None,
    fontsize=8,
    max_rows=6,
    min_rows=3,
    protein_coding_only=True,
    centered_names=False,
    arrowhead_length=None,
    arrowhead_length_proportion=None,
    arrowhead_width=None,
    arrow_length=None,
    all_same_line=False,
    show_arrows=True,
    bold_genes: list[str]=[],
    highlight_genes: list[str]=[],
    closest_to: int | None = None,
    
) -> None:
    """Plot a gene track (based on GENE_ANNOTATIONS) for a given range.

        plot_gene_track(region, ax=ax)

    Note: if too many genes are in the given range, then only a subset of genes will be plotted.
    (otherwise the track will be too crowded to read).
    """
    bold_genes = set(bold_genes)
    highlight_genes = set(highlight_genes)
    

    chr = region.chrom
    start = region.start
    end = region.end

    center = (start + end) // 2

    # Requires unprefixed chromosome
    # Get genes in gene regions (only protein-coding genes or IG genes)
    # If there's a gene filter, check through all genes
    # If not, then choose only protein-coding genes (if specified as True)
    candidates = GENE_ANNOTATIONS.genes_at_locus(
        contig=chr_unprefix(chr), position=start, end=end
    )
    if gene_filter:
        gene_filter = set([gene.upper() for gene in gene_filter])
        genes = list(filter(lambda g: g.gene_name in gene_filter, candidates))
    else:
        if protein_coding_only:
            genes = list(
                filter(
                    lambda g: g.biotype == "protein_coding",
                    candidates,
                )
            )
        else:
            genes = list(filter(lambda g: g.gene_name != "", candidates))

        # Narrow down the genes if there are too many to plot, keeping only a number of genes around the center
        # Keep genes whose midpoints are closest to the center
        if closest_to is None:
            genes.sort(key=lambda g: -1 if g.gene_name in bold_genes else 0 if g.gene_name in highlight_genes else min(abs(center - g.start), abs(center-g.end)))
        else:
            genes.sort(key=lambda g: -1 if g.gene_name in bold_genes else 0 if g.gene_name in highlight_genes else min(abs(closest_to-g.start), abs(closest_to-g.end)))
        # Add back any highlighted genes
        genes = genes[:max_rows]

    # Now sort genes by their start position
    genes.sort(key=lambda gene: gene.start)

    # Plot genes
    if ax is None:
        ax = plt.gca()

    # Prepare axes
    if vertical:
        ax.set_ylim(start, end)
        ax.invert_yaxis()
        ax.invert_xaxis()
    else:
        ax.set_xlim(start, end)

    if hide_axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines[["top", "right", "left", "bottom"]].set_visible(False)

    # Calculate values for plotting

    # Keep track of how many genes are plotted
    plot_counter = 0

    # Use 5 percent as a temporary padding value
    pct_5 = 5 * (end - start) / 100
    if arrowhead_length is None:
        if arrowhead_length_proportion is None:
            arrowhead_length = (end - start) / 75
        else:
            arrowhead_length = (end - start) * arrowhead_length_proportion
    plot_line_width = 0.4
    if arrowhead_width is None:
        arrowhead_width = 0.25
    if arrow_length is None:
        arrow_length = (end - start) / 50

    # Get genes which intersect the center directly
    direct_genes = GENE_ANNOTATIONS.genes_at_locus(
        contig=chr_unprefix(chr), position=center
    )
    direct_genes_set = set([gene.gene_name for gene in direct_genes])

    for gene in genes:

        exon_width_start = plot_counter - (plot_line_width / 2)
        exon_width = plot_line_width
        # Plot base arrow indicating strandness
        arr_start, full_length, arr_length = (
            (gene.start, gene.end - gene.start, arrow_length)
            if gene.strand == "+"
            else (gene.end, gene.start - gene.end, - arrow_length)
        )
        safe_start = max(start, gene.start) if gene.strand == "+" else (min(end, gene.end))
        ec, fc, ls = ("gray", "white", "solid") if (safe_start != arr_start) else ("black", "black", "solid")
        arr_axis_line = plot_counter+plot_line_width/2 + arrowhead_width/2
        if vertical:
            if gene.gene_name in highlight_genes:
                rect = Rectangle((plot_counter-0.5, gene.start), 1, gene.end-gene.start, fc="yellow", alpha=0.2, ec="none")
                ax.add_patch(rect)
            ax.vlines(plot_counter, gene.start, gene.end, colors="black")
        else:
            if gene.gene_name in highlight_genes:
                rect = Rectangle((gene.start, plot_counter-0.5), gene.end-gene.start, 1, fc="yellow", alpha=0.2, ec="none")
                ax.add_patch(rect)
            ax.hlines(plot_counter, gene.start, gene.end, colors="black")
        if show_arrows:
            if vertical:
                ax.hlines(arr_start, plot_counter, plot_counter+plot_line_width/2, colors="black", ls=":", lw=0.5)
                ax.hlines(arr_start, plot_counter+plot_line_width/2, arr_axis_line, colors="black", lw=0.5)
                ax.arrow(
                    arr_axis_line,
                    safe_start,
                    0,
                    arr_length,
                    head_width=arrowhead_width,
                    head_length=arrowhead_length,
                    length_includes_head=False,
                    lw=0.5,
                    ls=ls,
                    ec=ec,
                    fc=fc,
                )
            else:
                ax.vlines(arr_start, plot_counter, plot_counter+plot_line_width/2, colors="black", ls=":", lw=0.5)
                ax.vlines(arr_start, plot_counter+plot_line_width/2, arr_axis_line, colors="black", lw=0.5)
                ax.arrow(
                    safe_start,
                    arr_axis_line,
                    arr_length,
                    0,
                    head_width=arrowhead_width,
                    head_length=arrowhead_length,
                    length_includes_head=False,
                    lw=0.5,
                    ls=ls,
                    ec=ec,
                    fc=fc,
                )

        # Plot each exon as a rectangle on the gene line
        for exon in gene.exons:
            exon_length = exon.end - exon.start
            if vertical:
                ax.add_patch(
                    Rectangle(
                        (exon_width_start, exon.start),
                        exon_width,
                        exon_length,
                        edgecolor="black",
                        facecolor="black",
                    )
                )
            else:
                ax.add_patch(
                    Rectangle(
                        (exon.start, exon_width_start),
                        exon_length,
                        exon_width,
                        edgecolor="black",
                        facecolor="black",
                    )
                )

        # Get text position - and avoid putting the text out of axes
        # Bolden the text if passes directly through the center of the range (assuming a breakpoint is in the center)
        fontweight = "bold" if gene.gene_name in bold_genes else "normal"
        # fontweight = "normal"
        color = "black"
        if vertical:
            if centered_names:
                text_position = (gene.end + gene.start) / 2
                ha = "right"
                va = "center"
                if text_position + pct_5 > ax.get_ylim()[0]:
                    text_position = ax.get_ylim()[0]
                    va = "bottom"
                elif  text_position - pct_5 < ax.get_ylim()[1]:
                    text_position = ax.get_ylim()[1]
                    va = "top"
                text= ax.text(
                    plot_counter+plot_line_width/2+arrowhead_width,
                    text_position,
                    get_gene_name_or_id(gene),
                    ha=ha,
                    va=va,
                    fontsize=fontsize,
                    rotation=90,
                    color=color,
                    fontweight=fontweight,
                )
            else:
                row_position = plot_counter
                va = "bottom"
                text_position = gene.start - (pct_5 * 0.35)
                # if gene.strand == "-":
                    # text_position -= arrowhead_length
                if text_position - pct_5 < ax.get_ylim()[1]:
                    text_position = gene.end + (pct_5 * 0.25)
                    va = "top"
                    # if gene.strand == "+":
                        # text_position += arrowhead_length
                    if text_position + pct_5 > ax.get_ylim()[0]:
                        text_position = ax.get_ylim()[0]
                        va = "bottom"
                        row_position = plot_counter + plot_line_width / 2
                text = ax.text(
                    row_position,
                    text_position,
                    get_gene_name_or_id(gene),
                    ha="center",
                    va=va,
                    fontsize=fontsize,
                    rotation=90,
                    color=color,
                    fontweight=fontweight,
                )
                text.set_clip_on(True)
        else:
            if centered_names:
                text_position = (gene.end + gene.start) / 2
                ha = "center"
                va = "bottom"
                if text_position - pct_5 < ax.get_xlim()[0]:
                    text_position = ax.get_xlim()[0]
                    ha = "left"
                elif text_position + pct_5 > ax.get_xlim()[1]:
                    text_position = ax.get_xlim()[1]
                    ha = "right"
                text = ax.text(
                    text_position,
                    plot_counter+plot_line_width/2+arrowhead_width,
                    get_gene_name_or_id(gene),
                    ha=ha,
                    va=va,
                    fontsize=fontsize,
                    color=color,
                    fontweight=fontweight,
                )
            else:
                ha = "right"
                row_position = plot_counter
                text_position = gene.start - (pct_5 * 0.25)
                # if gene.strand == "-":
                    # text_position -= arrowhead_length
                if text_position - pct_5 < ax.get_xlim()[0]:
                    text_position = gene.end + (pct_5 * 0.25)
                    ha = "left"
                    # if gene.strand == "+":
                        # text_position += arrowhead_length
                    if text_position + pct_5 > ax.get_xlim()[1]:
                        text_position = ax.get_xlim()[1]
                        ha = "right"
                        row_position = plot_counter + plot_line_width / 2
                text = ax.text(
                    text_position,
                    row_position,
                    get_gene_name_or_id(gene),
                    ha=ha,
                    va="center",
                    fontsize=fontsize,
                    color=color,
                    fontweight=fontweight,
                )
                text.set_clip_on(True)

        # Highlight
        if gene.gene_name in highlight_genes:
            text.set_bbox(dict(facecolor='yellow', alpha=0.2, ec="none"))

        # Increment plot counter
        if not all_same_line:
            plot_counter += 1

    # Extend the axis limits if centered_gene_names is used
    if centered_names and plot_counter >= min_rows:
        if vertical:
            xmax, xmin = ax.get_xlim()
            ax.set_xlim(xmax + plot_line_width / 2, xmin)
        else:
            ymin, ymax = ax.get_ylim()
            ax.set_ylim(ymin, ymax + plot_line_width / 2)

    # If only a few genes were plotted, then add a bit of space padding to the plot
    if plot_counter < min_rows and not all_same_line:
        if vertical:
            ax.set_xlim(min_rows + plot_line_width + arrowhead_width/2, 0 - plot_line_width)
        else:
            ax.set_ylim(0 - plot_line_width, min_rows + plot_line_width + arrowhead_width/2)

 
    if region.reverse:
        if vertical:
            ax.invert_yaxis()
            for text in ax.findobj(matplotlib.text.Text):
                if text.get_text() != "":
                    match text.get_va():
                        case "top": 
                            text.set_ha("bottom")
                        case "bottom":
                            text.set_ha("top")
        else:
            ax.invert_xaxis()
            for text in ax.findobj(matplotlib.text.Text):
                if text.get_text() != "":
                    match text.get_ha():
                        case "left": 
                            text.set_ha("right")
                        case "right":
                            text.set_ha("left")




def plot_coverage_track(
    sample: ArimaPipelineSample,
    region: GenomicRegion,
    resolution: int,
    max_coverage=5,
    ax=None,
    hide_axes=False,
    vertical=False,
    fontsize=8,
    bar_color="#61B8D1",
    label="Coverage",
    label_fontsize=8,
    crosshairs: bool=False, 
    plotted_crosshairs: list[tuple[str, int, str]]=[],
    label_right=False,
) -> None:
    """Plot a coverage track for a given chromosome region.

        plot_coverage_track(sample, region, resolution, ax=ax)

    For best results, start and end should be multiples of the resolution.

    """
    chr = region.chrom
    start = region.start
    end = region.end

    # Get
    if ax is None:
        ax = plt.gca()

    # Get the coverage (VC) normalization vector
    zoom_data = sample.hic.getMatrixZoomData(
        chr_unprefix(chr), chr_unprefix(chr), "observed", "VC", "BP", resolution
    )
    # Position of norm vector is CHROM_INDEX + 1 (as the first stored chrom is the "ALL" chromosome)
    norm_position = CHROM_INDICES[chr] + 1
    norm_vector = zoom_data.getNormVector(norm_position)

    # Subset the norm vector for the given region
    norm_start = max(0, start // resolution)
    norm_end = (end + resolution) // resolution
    norm_vector = norm_vector[norm_start:norm_end]

    # Ensure norm vector is correct size
    if start < 0:
        norm_vector = np.pad(norm_vector, (-start // resolution, 0), "constant")
    if end > CHROM_SIZES[chr]:
        norm_vector = np.pad(
            norm_vector, (0, (end - CHROM_SIZES[chr]) // resolution), "constant"
        )

    # 0.5 offset correction as dealing with discrete bins
    positions = ((np.arange(norm_vector.size) + 0.5) * resolution) + start

    if vertical:
        ax.barh(positions, norm_vector, height=resolution, color=bar_color)
        ax.set_xlim(0, max_coverage)
        ax.set_ylim(start, end)
        ax.invert_yaxis()
        ax.invert_xaxis()
        ax.set_xticks([max_coverage, 0])
        ax.xaxis.tick_bottom()
        ax.get_xticklabels()[0].set_ha("left")
        ax.get_xticklabels()[-1].set_ha("right")
        ax.set_yticks([])
        ax.spines[["top", "left", "right"]].set_visible(False)
    else:
        ax.bar(positions, norm_vector, width=resolution, color=bar_color)
        ax.set_xlim(start, end)
        ax.set_ylim(0, max_coverage)
        ax.set_yticks([0, max_coverage])
        ax.get_yticklabels()[0].set_va("bottom")
        ax.get_yticklabels()[-1].set_va("top")
        ax.spines[["top", "right", "bottom"]].set_visible(False)
        ax.set_xticks([])
        
    if label_right:
        ax.yaxis.tick_right()
        ax.get_yticklabels()[0].set_va("bottom")
        ax.get_yticklabels()[-1].set_va("top")
        ax.spines[["left"]].set_visible(False)
        ax.spines[["right"]].set_visible(True)

    if hide_axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines[["top", "right", "left", "bottom"]].set_visible(False)

    # Add crosshairs if specified
    if crosshairs:
        for chr, pos, col, alpha in plotted_crosshairs:
            if chr == chr:
                if vertical:
                    ax.axhline(
                        pos,
                        color=col,
                        linestyle=(0, (1, 5)),
                        linewidth=1,
                        alpha=alpha,
                    )
                else:
                    ax.axvline(
                        pos,
                        color=col,
                        linestyle=(0, (1, 5)),
                        linewidth=1,
                        alpha=alpha,
                    )


    # Add label
    if vertical:
        ax.text(
            0,
            1,
            label,
            ha="left",
            va="top",
            transform=ax.transAxes,
            color="gray",
            fontdict={"fontsize": label_fontsize},
            rotation=90,
        )
    else:
        ax.text(
            0,
            1,
            label,
            ha="left",
            va="top",
            transform=ax.transAxes,
            color="gray",
            fontdict={"fontsize": label_fontsize},
        )
    if region.reverse:
        if vertical:
            ax.invert_yaxis()
        else:
            ax.invert_xaxis() 

    



def plot_bigwig_track(
    bw_handle: object,
    region: GenomicRegion,
    num_bins=1000,
    hide_axes=False,
    vertical=False,
    ax=None,
    fontsize=8,
    label: str = "",
    color="blue",
    ymax: float | None=None,
    crosshairs: bool=False, 
    plotted_crosshairs: list[tuple[str, int, str, float]]=[],
    label_right=False,
) -> None:

    chr = region.chrom
    start = region.start
    end = region.end

    # Check if chromosomes are prefixed or unprefixed
    if "chr1" in bw_handle.chroms().keys():
        pass
    else:
        chr = chr_unprefix(chr)

    # Get the data from the bigwig file
    # FIrst ensure bounds are safe 
    safe_start = max(0, start)
    safe_end = min(CHROM_SIZES[chr], end)
    safe_nbins = (safe_end - safe_start) // ((end - start) // num_bins)
    data = bw_handle.stats(chr, safe_start, safe_end, type="mean", nBins=safe_nbins, numpy=True)
    # Pad with extra zeros if was out of bounds initially
    bin_width = (end - start) / num_bins
    if start < 0:
        data = np.pad(data, (int(-start // bin_width), 0), "constant")
    if end > CHROM_SIZES[chr]:
        data = np.pad(data, (0, int((end - CHROM_SIZES[chr]) // bin_width)+1), "constant")

    # TODO: Using an arbitrary ymax for now, but probably want to choose a different normalization at some point
    if ymax is None:
        ymax = np.sqrt(bw_handle.header()["maxVal"])
    ymax_label = f"{ymax:.1f}" if ymax > 1 else f"{ymax:.2f}"

    positions = np.linspace(start, end, num_bins)

    # Plot the data depending on if horizontal or vertical axis
    if ax is None:
        ax = plt.gca()

    if vertical:
        # Fill from right to data peak on left
        ax.fill_betweenx(positions, np.zeros(num_bins), data, color=color)
        ax.set_ylim(start, end)
        ax.invert_yaxis()
        ax.set_xlim(0, ymax)
        ax.invert_xaxis()
        ax.spines[["top", "right", "left", ]].set_visible(False)
        ax.set_xticks([ymax, 0], [ymax_label, "0"])
        ax.get_xticklabels()[0].set_ha("left")
        ax.get_xticklabels()[-1].set_ha("right")
        ax.set_yticks([])
    else:
        ax.fill_between(positions, np.zeros(num_bins), data, color=color)
        ax.set_xlim(start, end)
        ax.set_ylim(0, ymax)
        ax.set_yticks([0, ymax], ["0", ymax_label])
        ax.set_xticks([])
        if label_right:
            ax.spines[["top", "left", "bottom"]].set_visible(False)
            ax.yaxis.tick_right()
        else:
            ax.spines[["top", "right", "bottom"]].set_visible(False)
        ax.get_yticklabels()[0].set_va("bottom")
        ax.get_yticklabels()[-1].set_va("top")

    if hide_axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines[["top", "right", "left", "bottom"]].set_visible(False)

    # Add crosshairs if specified
    if crosshairs:
        for chr, pos, col, alpha in plotted_crosshairs:
            if chr == chr:
                if vertical:
                    ax.axhline(
                        pos,
                        color=col,
                        linestyle=(0, (1, 5)),
                        linewidth=1,
                        alpha=alpha,
                    )
                else:
                    ax.axvline(
                        pos,
                        color=col,
                        linestyle=(0, (1, 5)),
                        linewidth=1,
                        alpha=alpha,
                    )


    # Add label
    if vertical:
        ax.text(
            0,
            1,
            label,
            ha="left",
            va="top",
            transform=ax.transAxes,
            color="gray",
            fontdict={"fontsize": fontsize},
            rotation=90,
        )
    else:
        ax.text(
            0,
            1,
            label,
            ha="left",
            va="top",
            transform=ax.transAxes,
            color="gray",
            fontdict={"fontsize": fontsize},
        )

    if region.reverse:
        if vertical:
            ax.invert_yaxis()
        else:
            ax.invert_xaxis()

    return ax



def plot_arrow_track(region: GenomicRegion, label_chr=True, ax=None) -> None:
    """Plot simple chromosome arrows depicting the bounds.

        plot_arrow_track(region, ax=ax)
        
    """

    if ax is None:
        ax = plt.gca()
    chrom, start, end = region.chrom, region.start, region.end
    arrowhead_style = "left"

    fig = plt.gcf()
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width, height = bbox.width, bbox.height
    head_length = region.get_size() * (height / width) / 2
    
    # head_length = ((end - start) / 25)
    
    arrow_end = end
    arrow_start = start
    ax.arrow(arrow_start, 0.5, arrow_end-arrow_start, 0, ec="none", fc=CHROM_COLORS[chrom], width=0.5, head_length=head_length, head_width=1, shape=arrowhead_style, length_includes_head=True)
    ax.set_xlim(start, end)
    if label_chr:
        ax.set_ylim(-0.5, 0.5)
        ax.text(start, 0, f"{start:,}", ha="left", va="top")
        ax.text(end, 0, f"{end:,}", ha="right", va="top")
        ax.text((start + end) / 2, 0, f"{chrom}", ha="center", va="top")
        # If is minus strand, flip all ha of text
    if region.reverse:
        ax.invert_xaxis()
        for text in ax.texts:
            if text.get_ha() == "left":
                text.set_ha("right")
            elif text.get_ha() == "right":
                text.set_ha("left")
    blank_axis(ax)



def plot_ctcf_track(bedfile: str, region: GenomicRegion, ax=plt.Axes | None,s=20, vertical=False, colorA="seagreen", colorB="seagreen"):
    """Plot oriented CTCF motifs (given a bedfile with columns chr, start, end, strand)

        plot_oriented_ctcf_track(bedfile, region, ax=ax)
    """

    if ax is None:
        ax = plt.gca()

    chr, start, end = region.get_unpacked()
        
    motifs = pd.read_csv(bedfile, sep="\s+", header=None, names=["chr", "start", "end", "strand"])
    motifs = motifs[motifs["chr"] == chr]
    motifs = motifs[(motifs["start"] >= start) & (motifs["start"] <= end) | (motifs["end"] >= start) & (motifs["end"] <= end)]
    
    pos = motifs[motifs["strand"] == "+"].start
    neg = motifs[motifs["strand"] == "-"].end

    if vertical:
        marker_pos = "v"
        marker_neg = "^"
        # Pre-emptively reverse markers if region is reversed
        if region.reverse:
            marker_pos, marker_neg = marker_neg, marker_pos
        ax.scatter([0.5] * len(pos), pos, color=colorA, s=s, marker=marker_pos)
        ax.scatter([0.5] * len(neg), neg, color=colorB, s=s, marker=marker_neg)
        ax.set_ylim(start, end)
        ax.invert_yaxis()
        ax.set_xlim(1, 0)
    else:
        marker_pos = ">"
        marker_neg = "<"
        # Pre-emptively reverse markers if region is reversed
        if region.reverse:
            marker_pos, marker_neg = marker_neg, marker_pos
        ax.scatter(pos, [0.5] * len(pos), color=colorA, s=s, marker=marker_pos)
        ax.scatter(neg, [0.5] * len(neg), color=colorB, s=s, marker=marker_neg)
        ax.set_xlim(start, end)
        ax.set_ylim(0, 1)

    if region.reverse:
        if vertical:
            ax.invert_yaxis()
        else:
            ax.invert_xaxis()

    blank_axis(ax)

def plot_cnv_track(
        cnv_profile_path: str,
        cnv_segment_path: str,
        chr: str,
        ax=None,
        vertical=False,
        cnv_lim: tuple[float,float]=None,
        linewidth=3,
        dot_alpha=0.5,
        dot_size=0.5,
        plot_scatter=False,
        plot_segments=True,
        cnv_res=25000,
        locus_lim: tuple[int, int]=None,
        show_zero_line=False,
):

    # Read
    columns = ["chr", "start", "end", "value"]
    df_profile = pd.read_csv(cnv_profile_path, sep="\s+", names=columns)
    df_segment = pd.read_csv(cnv_segment_path, sep="\s+", names=columns)

    subset_profile = df_profile[(df_profile.chr == chr)]
    subset_segment = df_segment[(df_segment.chr == chr)]

    # Plot scatter plot of subset_profile
    positions = subset_profile[subset_profile.value>0].start
    cnv_values = np.log2(subset_profile[subset_profile.value>0].value)

    cmin = cnv_values.min()
    cmax = cnv_values.max()
    cabs = max(abs(cmin), abs(cmax))
    cmap = LinearSegmentedColormap.from_list('rg',["red", "gray", "green"], N=256)
    cvalues = [ cmap((v+2) / 4) for v in cnv_values]

    if ax is None:
        ax = plt.gca()

    if show_zero_line:
        ax.axhline(0, ls=":", color="gray")

    if plot_scatter:
        if vertical:
            ax.scatter(cnv_values, positions, alpha=dot_alpha, c=cvalues, marker=".", s=dot_size, rasterized=True)
        else:
            ax.scatter(positions, cnv_values, alpha=dot_alpha, c=cvalues, marker=".", s=dot_size, rasterized=True)

    if plot_segments:
        minlen = 1
        res = cnv_res
        for _, (_, start, end, value) in subset_segment.iterrows():
            si = start // res
            ei = end // res
            if ei - si >= minlen:
                tmp = subset_profile.value[si:ei]
                mask = tmp == 0
                zero_ratio = mask.sum() / mask.size
                if zero_ratio > 0.80:
                    continue
                seg_cnv_value = np.log2(np.median(tmp[tmp!=0]))
                color=cmap((seg_cnv_value + 2)/4)
                if vertical:
                    ax.vlines(seg_cnv_value, start, end, color=color, linewidth=linewidth)
                else:
                    ax.hlines(seg_cnv_value, start, end, color=color, linewidth=linewidth)

    if vertical:
        if locus_lim is not None:
            ax.set_ylim(*locus_lim)
        else:
            ax.set_ylim(0, CHROM_SIZES[chr])
        ax.set_yticks([])
        ax.set_xlabel("$\log_2$(CN)")
        if cnv_lim is not None:
            ax.set_xlim(*cnv_lim)
    else:
        if locus_lim is not None:
            ax.set_xlim(*locus_lim)
        else:
            ax.set_xlim(0, CHROM_SIZES[chr])
        ax.set_xticks([])
        ax.set_ylabel("$\log_2$(CN)")
        if cnv_lim is not None:
            ax.set_ylim(*cnv_lim)

    if vertical:
        ax.invert_yaxis()
        ax.invert_xaxis()

    return ax


def plot_assembled_triangle(assembled: AssembledHic, resolution: int, ax: plt.Axes=None, vmax: float | None =None, aspect="equal", 
                            rasterized=False, cmap=REDMAP, normalization="?", plot_points: list[PairedRegion]=[], neoloop_lw=1, neoloop_ls="-", 
                           show_arcs=False):

    data = assembled.data

    for region in assembled.plot_regions:
        if region.genomic_region is not None:
            assert region.genomic_region.bin_align_res == resolution 
            assert region.genomic_region.bin_align_res == resolution

    if ax is None:
        ax = plt.gca()

    dim = data.shape[0]
    col, row = np.meshgrid(np.arange(dim), np.arange(dim))
    xy_offsets = np.stack([col.flatten(), row.flatten()]).T
    
    # Invert the y axis
    xy_offsets[:,1] = dim - xy_offsets[:,1] 
    
    # Calculate offsets with 45 degree rotation
    cos45 = np.cos(np.pi/4)
    affine = matplotlib.transforms.Affine2D().translate(0, -data.shape[0]).rotate_around(0, 0, np.pi/4).scale(cos45, cos45).translate(0.5, 0)
    xy_offsets = affine.transform(xy_offsets)
    
    col = xy_offsets[:, 0].reshape((dim, dim))
    row = xy_offsets[:, 1].reshape((dim, dim))

    vmax = np.nanmax(data) if vmax is None else vmax
    # cmap = cmap.copy()
    # cmap.set_under("#eee")
    im = ax.pcolormesh(col, row, data, cmap=cmap, vmax=vmax, vmin=0, rasterized=rasterized)

    # Calculate plot points
    def align_plot_point(p: PairedRegion):
        mat_pos = []
        cum_bins = 0 
        for plot_region in assembled.plot_regions:
            bin_start, bin_end = plot_region.bin_range
            if plot_region.genomic_region is not None:
                s = plot_region.genomic_region
                if s.overlaps(p.regionA):
                    new_pos = (p.regionA.get_center() - s.start) / (s.end + resolution - s.start)
                    new_pos = 1 - new_pos if s.reverse else new_pos 
                    new_pos = new_pos * (bin_end - bin_start) + cum_bins
                    if new_pos > 0:
                        mat_pos.append(new_pos)
                if s.overlaps(p.regionB):
                    new_pos = (p.regionB.get_center() - s.start) / (s.end + resolution - s.start)
                    new_pos = 1 - new_pos if s.reverse else new_pos 
                    new_pos = new_pos * (bin_end - bin_start) + cum_bins
                    if new_pos > 0:
                        mat_pos.append(new_pos)
            cum_bins +=  bin_end - bin_start
        if len(mat_pos) != 2:
            print(p, "failed", mat_pos)
            return None
        return data.shape[0] - np.min(mat_pos), np.max(mat_pos)
                    
    aligned_plot_points = []
    plot_point_radii = []
    if len(plot_points) > 0:
        for point in plot_points: 
            aligned = align_plot_point(point)
            if aligned is not None:
                py, px = aligned
                aligned_plot_points.append((px, py))
                plot_point_radii.append(point.regionA.get_size())

    # Add plot points
    if len(aligned_plot_points) > 0:
        # print(aligned_plot_points)
        transformed_aligned_plot_points = affine.transform(np.array(aligned_plot_points))
        for (x, y) in transformed_aligned_plot_points:
            ellipse_radius = dim/20
            ellipse = matplotlib.patches.Ellipse((x, y), ellipse_radius, ellipse_radius, fc="none", ec=LOOP_COLOUR, ls=neoloop_ls, lw=neoloop_lw)
            ax.add_patch(ellipse)
            
    # Add arcs if requested
    if show_arcs:
        for (px, dpy),pradius in zip(aligned_plot_points, plot_point_radii):
            py = data.shape[0] - dpy
            square_size = 1
            rect_width = max(pradius/ resolution * square_size, 1)
            halfsize= rect_width/2
            arc_y = -(square_size / 2)
            ax.add_patch(Rectangle((px-halfsize, arc_y), rect_width, square_size/2, fc=LOOP_COLOUR, ec=LOOP_COLOUR, clip_on=False))
            ax.add_patch(Rectangle((py-halfsize, arc_y), rect_width, square_size/2, fc=LOOP_COLOUR, ec=LOOP_COLOUR, clip_on=False))
            midarc = (px + py) / 2
            arc = matplotlib.patches.Arc((midarc, arc_y), abs(py-px), data.shape[1] / 20, theta1=180, theta2=360, ec=LOOP_COLOUR, clip_on=False, ls=neoloop_ls, lw=neoloop_lw)
            ax.add_patch(arc)

    # Set axis limits
    ax.set_xlim(0, data.shape[0])
    ax.set_ylim(0, (data.shape[1])/2)
    ax.set_aspect(aspect)

    # Add anchored size bar in top right corner of plot
    scalebar = AnchoredSizeBar(
        ax.transData,
        0.5, int_to_resolution(resolution), 'upper right', 
        pad=0.1,
        color='black',
        frameon=False,
        size_vertical=0.5,
    )
    ax.add_artist(scalebar)

    cax = ax.inset_axes([0.85, 0.85, 0.05, 0.03])
    vmax_label = str(int(round(vmax))) if (isinstance(vmax, int) or abs(vmax-round(vmax)) < 0.001) else f"{vmax:.2f}" if vmax < 1 else f"{vmax:.1f}"
    balancing = SHORT_NORM[normalization]
    plt.colorbar(im, cax=cax, orientation="horizontal", )
    cax.set_xticks([])
    cax.set_title(f"{vmax_label} ({balancing})", x=3, y=0.42, ha="right", va="center", transform=cax.transAxes, fontsize=8)

    # Only considering two plot regions for now - if there are more, then this won't work!
    for i, plot_region_left in enumerate(assembled.plot_regions):
        if plot_region_left.genomic_region is not None:
            left_start, left_end = plot_region_left.bin_range
            triangle_point = (left_end - left_start) / 2
            polygon = matplotlib.patches.Polygon([(left_start, 0), (left_end, 0), (left_start + triangle_point, triangle_point)], closed=True, ec="gray", fc="none", lw=0.5)
            ax.add_patch(polygon)
            for plot_region_right in assembled.plot_regions[i:]:
                if plot_region_right.genomic_region is not None:
                    right_start, right_end = plot_region_right.bin_range
                    distance = right_start - left_end
                    bottom_corner_x = left_end + distance / 2
                    bottom_corner_y = distance / 2
                    bottom_corner = (bottom_corner_x, bottom_corner_y)
                    d_left = (left_end - left_start) / 2
                    d_right = (right_end - right_start) / 2
                    left_corner_x = bottom_corner_x - d_left
                    left_corner_y = bottom_corner_y + d_left
                    left_corner = (left_corner_x, left_corner_y)
                    right_corner_x = bottom_corner_x + d_right
                    right_corner_y = bottom_corner_y + d_right
                    right_corner = (right_corner_x, right_corner_y)
                    top_corner = (bottom_corner_x - d_left + d_right, bottom_corner_y + d_left + d_right)
                    corners = [bottom_corner, left_corner, top_corner, right_corner]
                    polygon = matplotlib.patches.Polygon(corners, closed=True, ec="gray", fc="none", lw=0.5)
                    ax.add_patch(polygon)
    
    blank_axis(ax)

def plot_qc(sample: ArimaPipelineSample, figsize=(13, 8)) -> plt.Figure:

    fig = plt.figure(figsize=figsize)
    qc = sample.qc

    # Plot of unique valid pairs vs all raw pairs
    plt.subplot(5, 1, 1)
    raw_pairs = int(qc.raw_pairs) / 1e6
    unique_valid_pairs = int(qc.unique_valid_pairs) / 1e6
    plt.barh(1, raw_pairs, color="lightgray")
    plt.barh(1, unique_valid_pairs, color="black")
    plt.yticks([])
    plt.xlim(0, max(600, raw_pairs))
    plt.ylim(0.5, 4)
    plt.text(0, 2, "Total informative pairs (million pairs)", fontsize=12, va="bottom")
    plt.gca().spines[["top", "left", "right"]].set_visible(False)
    raw_pair_ha = "right" if raw_pairs > 500 else "left"
    raw_pair_x = raw_pairs - 5 if raw_pairs > 500 else raw_pairs+3
    plt.text(raw_pair_x, 1, f"{int(raw_pairs)} million raw pairs", color="black", ha=raw_pair_ha, va="center", fontsize=9)
    plt.text(0, 1.5, f"{int(unique_valid_pairs)} million unique valid pairs", color="black", ha="left", va="bottom", fontsize=9)
    plt.title(f"Arima SV Pipeline QC Metrics for {sample.id}")


    # Plot of raw and mapped reads
    plt.subplot(5, 1, 2)
    raw_pairs = int(qc.raw_pairs) / 1e6
    mapped_se = int(qc.mapped_se_reads) / 1e6
    mapped_se_pct = int(qc.mapped_se_reads_pct)
    plt.barh(2, raw_pairs * 2, color="deepskyblue")
    plt.barh(1, mapped_se, color="lightskyblue")
    plt.yticks([])
    plt.xlim([0, max(1200, raw_pairs*2)])
    plt.ylim(0.5, 4)
    plt.text(0, 2.5, "Read mapping (single-end mode, million reads)", va="bottom", fontsize=12)
    # plt.xlabel("Count (million reads)")
    plt.gca().spines[["top", "left", "right"]].set_visible(False)

    patches = plt.gca().patches
    labels = [
        f"{raw_pairs:.0f} million pairs = {raw_pairs*2:.0f} million total SE reads ",
        f"{mapped_se:.0f} million mapped SE reads ({mapped_se_pct}%)",
    ]
    for patch, label in zip(patches, labels):
        plt.text(
            10,
            patch.get_y() + patch.get_height() / 2,
            label,
            color="black",
            ha="left",
            va="center",
            fontsize=9,
        )

    # Plot of valid, duplicate and invalid reads (and invalid composition)
    plt.subplot(5, 1, 3)
    left = 0
    unique_valid_pairs = qc.unique_valid_pairs_pct
    duplicates = qc.duplicated_pct
    invalid = qc.invalid_pct
    plt.barh(1, invalid, color="crimson", left=left)
    plt.barh(1, duplicates, color="lightgray", left=left + invalid)
    plt.barh(1, unique_valid_pairs, color="limegreen", left=left + invalid + duplicates)

    left = 0
    circular = qc.same_circular_pct
    dangling = qc.same_dangling_pct
    fragment = qc.same_fragment_internal_pct
    re_ligation = qc.re_ligation_pct
    contiguous = qc.contiguous_pct
    wrong_size = qc.wrong_size_pct
    base_color = "crimson"
    plt.barh(0, circular, color=base_color, left=left, alpha=6 / 6)
    plt.barh(0, dangling, color=base_color, left=left + circular, alpha=5 / 6)
    plt.barh(
        0, fragment, color=base_color, left=left + circular + dangling, alpha=4 / 6
    )
    plt.barh(
        0,
        re_ligation,
        color=base_color,
        left=left + circular + dangling + fragment,
        alpha=3 / 6,
    )
    plt.barh(
        0,
        contiguous,
        color=base_color,
        left=left + circular + dangling + fragment + re_ligation,
        alpha=2 / 6,
    )
    plt.barh(
        0,
        wrong_size,
        color=base_color,
        left=left + circular + dangling + fragment + re_ligation + contiguous,
        alpha=1 / 6,
    )

    plt.yticks([])
    plt.xlim([0, 100])
    plt.ylim(-0.5, 3)
    plt.text(0, 1.5, "Pair validity (% of aligned pairs)", fontsize=12, va="bottom")
    # plt.xlabel("% of pairs")
    plt.gca().spines[["top", "left", "right"]].set_visible(False)

    patches = plt.gca().patches
    labels = [
        f"{invalid}% invalid",
        f"{duplicates}% dups",
        f"{unique_valid_pairs}% valid",
        f"{circular}% circular",
        f"{dangling}% dangling",
        f"{fragment}% internal",
        f"{re_ligation}% re-ligated",
        f"{contiguous}% contiguous",
        f"{wrong_size}% wrong size",
    ]
    for patch, label in zip(patches, labels):
        if patch.get_y() > 0:
            if patch.get_width() > 10:
                plt.text(
                    patch.get_x() + 0.5,
                    patch.get_y() + patch.get_height() / 2,
                    label,
                    color="black",
                    ha="left",
                    va="center",
                    fontsize=9,
                )
            elif patch.get_width() > 4:
                plt.text(
                    patch.get_x() + 0.5,
                    patch.get_y() + patch.get_height() / 2,
                    label,
                    color="black",
                    ha="left",
                    va="center",
                    fontsize=6,
                )
        else:
            if patch.get_width() > 5:
                plt.text(
                    patch.get_x() + 0.5,
                    patch.get_y() + patch.get_height() / 2,
                    label,
                    color="black",
                    ha="left",
                    va="center",
                    fontsize=7,
                )
            else:
                pass

    plt.subplot(5, 6, 19)
    wedges = plt.pie([circular, 100-circular], colors=[base_color, "lightgray"], radius=0.5)
    plt.text(0, 0.8, f"Circular: {circular}%", fontsize=10, ha="center", va="center")
    plt.ylim(-0.5, 1.1)

    plt.subplot(5, 6, 20)
    wedges = plt.pie([dangling, 100-dangling], colors=[base_color, "lightgray"], radius=0.5)
    wedges[0][0].set_alpha(5/6)
    plt.text(0, 0.8, f"Dangling: {dangling}%", fontsize=10, ha="center", va="center")
    plt.ylim(-0.5, 1.1)

    plt.subplot(5, 6, 21)
    wedges = plt.pie([fragment, 100-fragment], colors=[base_color, "lightgray"], radius=0.5)
    wedges[0][0].set_alpha(4/6)
    plt.text(0, 0.8, f"Internal: {fragment}%", fontsize=10, ha="center", va="center")
    plt.ylim(-0.5, 1.1)

    plt.subplot(5, 6, 22)
    wedges = plt.pie([re_ligation, 100-re_ligation], colors=[base_color, "lightgray"], radius=0.5)
    wedges[0][0].set_alpha(3/6)
    plt.text(0, 0.8, f"Re-ligated: {re_ligation}%", fontsize=10, ha="center", va="center")
    plt.ylim(-0.5, 1.1)

    plt.subplot(5, 6, 23)
    wedges = plt.pie([contiguous, 100-contiguous], colors=[base_color, "lightgray"], radius=0.5)
    wedges[0][0].set_alpha(2/6)
    plt.text(0, 0.8, f"Contiguous: {contiguous}%", fontsize=10, ha="center", va="center")
    plt.ylim(-0.5, 1.1)

    plt.subplot(5, 6, 24)
    wedges = plt.pie([wrong_size, 100-wrong_size], colors=[base_color, "lightgray"], radius=0.5)
    wedges[0][0].set_alpha(1/6)
    plt.text(0, 0.8, f"Wrong size: {wrong_size}%", fontsize=10, ha="center", va="center")
    plt.ylim(-0.5, 1.1)

    # Plot of library size
    plt.subplot(5, 5, 21)
    mean_lib_length = int(qc.mean_lib_length)
    plt.barh(0, mean_lib_length, color="black")
    plt.xlim([0, 400])
    plt.yticks([])
    plt.ylim(-0.5, 1)
    plt.text(0, 0.5, f"Mean lib length\n{mean_lib_length}bp", fontsize=11, va="bottom")
    plt.gca().spines[['top', 'left', 'right']].set_visible(False)

    # Plot of % truncated
    plt.subplot(5, 5, 22)
    truncated = qc.truncated_pct
    plt.barh(0, truncated, color="darkorange")
    plt.xlim([0, 100])
    plt.ylim(-0.5, 1)
    plt.text(0, 0.5, f"% truncated\n{truncated}%", fontsize=11, va="bottom")
    plt.yticks([])
    plt.gca().spines[['top', 'left', 'right']].set_visible(False)

    # Plot of intra and inter
    plt.subplot(5, 5, 23)
    left = 0
    intra = qc.intra_pairs_pct
    inter = qc.inter_pairs_pct
    plt.barh(0, intra, color="purple", left=left)
    plt.barh(0, inter, color="plum", left=left + intra)
    plt.yticks([])
    plt.ylim(-0.5, 1)
    plt.text(0, 0.5, f"% intra/inter\n{intra}%/{inter}%", fontsize=11, va="bottom")
    plt.xlim([0, 100])
    plt.gca().spines[['top', 'left', 'right']].set_visible(False)

    # Plot of LCIS and trans
    plt.subplot(5, 5, 24)
    lcis_trans_ratio = qc.lcis_trans_ratio
    plt.barh(0, lcis_trans_ratio, color="slateblue")
    plt.xlim([0, 4])
    plt.yticks([])
    plt.ylim(-0.5, 1)
    plt.text(0, 0.5, f"Lcis/Trans ratio\n{lcis_trans_ratio}", fontsize=11, va="bottom")
    plt.gca().spines[['top', 'left', 'right']].set_visible(False)

    # # Plot of Number of SV Breakfinder Calls
    plt.subplot(5, 5, 25)
    num_sv_calls = len(sample.breakpoints)
    plt.barh(0, num_sv_calls, color="dimgray")
    plt.xlim([0, 100])
    plt.yticks([])
    plt.ylim(-0.5, 1)
    plt.text(0, 0.5, f"Breakpoints\n{num_sv_calls}", fontsize=11, va="bottom")
    plt.gca().spines[['top', 'left', 'right']].set_visible(False)

    
    return fig



def plot_composite_context_and_zoom(
    sample: ArimaPipelineSample,
    bpoint: Breakpoint,
    figsize=(11, 6.3),
    zoom_resolution=10000,
    zoom_radius=300000,
    gene_filter=None,
    title=None,
    title_fontsize=8,
    title_ha="left",
    gene_fontsize=7,
    # extra_bedpe: list[BedpeLine] = [],
    coverage_track=True,
    hide_track_axes=True,
    extra_bigwig_handles: list[tuple[str, object]] = [],
    crosshairs=False,
    grid=False,
    plot_at_bpoint_resolution=False,
    capped_resolution=None,
    min_gene_rows=3,
    centered_gene_names=False,
    keygenes: list[str]=[],
    **kwargs,
) -> plt.Figure:
    """Plot whole-chromosome context on left and zoomed breakfinder bpoint on right with gene track."""

    # Get figure and separate out axes
    fig = plt.figure(figsize=figsize, constrained_layout=True)

    # Calculate the number of rows and columns
    num_bigwig_tracks = len(extra_bigwig_handles)
    sizes_bigwig = [0.5] * num_bigwig_tracks
    num_coverage_tracks = 1 if coverage_track else 0
    sizes_coverage = [0.5] * num_coverage_tracks

    num_rows = 2 + num_coverage_tracks + num_bigwig_tracks
    num_cols = 3 + num_coverage_tracks + num_bigwig_tracks
    height_ratios = sizes_coverage + sizes_bigwig + [1.5, 8]
    width_ratios = [sum(height_ratios)] + sizes_coverage + sizes_bigwig + [1.5, 8]

    spec = GridSpec(
        ncols=num_cols,
        nrows=num_rows,
        figure=fig,
        height_ratios=height_ratios,
        width_ratios=width_ratios,
        wspace=0,
        hspace=0,
    )

    # Get axis handles
    ax_large = fig.add_subplot(spec[:, 0])

    ax_zoom_row = 1 + num_coverage_tracks + num_bigwig_tracks
    ax_zoom_col = 2 + num_coverage_tracks + num_bigwig_tracks
    ax_zoom = fig.add_subplot(spec[ax_zoom_row, ax_zoom_col])

    ax_bigwig_horizontal_start = 1 if coverage_track else 0
    ax_bigwig_horizontal_handles = [
        fig.add_subplot(spec[start, ax_zoom_col])
        for start in range(
            ax_bigwig_horizontal_start, ax_bigwig_horizontal_start + num_bigwig_tracks
        )
    ]

    ax_bigwig_vertical_start = 2 if coverage_track else 1
    ax_bigwig_vertical_handles = [
        fig.add_subplot(spec[ax_zoom_row, start])
        for start in range(
            ax_bigwig_vertical_start, ax_bigwig_vertical_start + num_bigwig_tracks
        )
    ]

    ax_genes_top = fig.add_subplot(spec[ax_zoom_row - 1, ax_zoom_col])
    ax_genes_left = fig.add_subplot(spec[ax_zoom_row, ax_zoom_col - 1])

    # Unpack breakfinder bpoint
    chrA, posA = bpoint.breakendA.chrom, bpoint.breakendA.pos
    chrB, posB = bpoint.breakendB.chrom, bpoint.breakendB.pos

    # Plot zoomed hic matrix first to get axis bounds
    if plot_at_bpoint_resolution:
        if capped_resolution is not None:
            zoom_resolution = max(bpoint.resolution, capped_resolution)
            zoom_radius = 30 * zoom_resolution
        else:
            zoom_resolution = bpoint.resolution
            zoom_radius = 30 * zoom_resolution

    regionX, regionY = bpoint.get_centered_regions_with_radius(radius=zoom_radius, bin_align_res=zoom_resolution)
    involved_genes = [g.gene_name for g in bpoint.breakendA.get_intersecting_genes() + bpoint.breakendB.get_intersecting_genes()]

    plot_hic_region_matrix(
        sample, 
        regionX, 
        regionY,  
        resolution=zoom_resolution, 
        measure="observed",
        normalization="NONE",
        ax=ax_zoom, 
        show_breakpoints=True,
        crosshairs=crosshairs, 
        grid=grid, 
        breakpoint_highlight=bpoint, 
    )

    # Choose a chromosome context resolution
    if chrA == chrB:
        if CHROM_SIZES[chrA] < 100000000:
            context_resolution = 250000
        else:
            context_resolution = 500000
    else:
        if CHROM_SIZES[chrA] + CHROM_SIZES[chrB] < 200000000:
            context_resolution = 500000
        else:
            context_resolution = 1000000

    # Plot chromosome context

    plot_hic_chr_context(
        sample,
        chrA,
        chrB,
        context_resolution,
        show_breakpoints=True,
        region_highlight=(regionX, regionY),
        measure="observed",
        normalization="NONE",
        norm_constant_normalize=False,
        ax=ax_large,
    )

    # Plot coverage tracks
    if coverage_track:
        ax_coverage_top = fig.add_subplot(spec[0, ax_zoom_col])
        ax_coverage_bottom = fig.add_subplot(spec[ax_zoom_row, 1])
        plot_coverage_track(
            sample,
            regionX,
            zoom_resolution,
            ax=ax_coverage_top,
            label_fontsize=7,
            crosshairs=crosshairs, 
            label_right=True,
        )
        plot_coverage_track(
            sample,
            regionY,
            zoom_resolution,
            vertical=True,
            ax=ax_coverage_bottom,
            label_fontsize=7,
            crosshairs=crosshairs, 
        )

    # Plot gene tracks
    plot_gene_track(
        regionX,
        ax=ax_genes_top,
        fontsize=gene_fontsize,
        gene_filter=gene_filter,
        hide_axes=hide_track_axes,
        min_rows=min_gene_rows,
        centered_names=centered_gene_names,
        arrow_length = (regionX.end - regionX.start)  / 150,
        arrowhead_width=0.5,
        highlight_genes=keygenes,
        bold_genes=involved_genes,
    )
    plot_gene_track(
        regionY,
        ax=ax_genes_left,
        vertical=True,
        fontsize=gene_fontsize,
        gene_filter=gene_filter,
        hide_axes=hide_track_axes,
        # crosshairs=crosshairs, 
        min_rows=min_gene_rows,
        centered_names=centered_gene_names,
        arrow_length = (regionY.end - regionY.start)  / 150,
        arrowhead_width=0.5,
        highlight_genes=keygenes,
        bold_genes=involved_genes,
    )

    # For each bigwig track, plot
    for (i, (label, bw_handle)), ax_horizontal, ax_vertical, color in zip(
        enumerate(extra_bigwig_handles),
        ax_bigwig_horizontal_handles,
        ax_bigwig_vertical_handles,
        BIGWIG_COLORS,
    ):
        plot_bigwig_track(
            bw_handle,
            regionX,
            label=label,
            ax=ax_horizontal,
            color=color,
            fontsize=7,
            crosshairs=crosshairs, 
            label_right=True,
            # plotted_crosshairs=plotted_crosshairs,
        )
        plot_bigwig_track(
            bw_handle,
            regionY,
            label=label,
            ax=ax_vertical,
            vertical=True,
            color=color,
            fontsize=7,
            crosshairs=crosshairs, 
            # plotted_crosshairs=plotted_crosshairs,
        )

    # If no specified title, then make metadata title
    if title is None:
        title = f"Sample={sample.id}\nZoomCenterX={chrA}:{regionX.get_center()}, ZoomCenterY={chrB}:{regionY.get_center()}\nZoomBoundsX={regionX}, ZoomBoundsY={regionY}\nZoomRes={zoom_resolution}bp, ZoomRadius={zoom_radius}bp"
    if title_ha == "left":
        fig.suptitle(title, fontsize=title_fontsize, x=0.02, ha="left")
    else:
        fig.suptitle(title, fontsize=title_fontsize)

    return fig


def plot_composite_multires_breakpoint(
    sample: ArimaPipelineSample,
    bpoint: Breakpoint,
    figheight=1.8,
    default_zoom_resolution=10000,
    title=None,
    title_fontsize=8,
) -> plt.Figure: 
    """Plot a single breakpoint at multiple resolutions in a row."""

    resolutions = [1000000, 500000, 100000, 50000, 10000, 5000, 1000]
    num_plots = len(resolutions)
    fig, ax = plt.subplots(1, num_plots, figsize=(num_plots * figheight, figheight*0.9))
    posA = bpoint.breakendA.pos
    posB = bpoint.breakendB.pos

    for i, resolution in enumerate(resolutions):
        regionX, regionY = bpoint.get_centered_regions_with_radius(radius=25*resolution, bin_align_res=resolution)
        chrA, posA = bpoint.breakendA.chrom, bpoint.breakendA.pos
        chrB, posB = bpoint.breakendB.chrom, bpoint.breakendB.pos
        zoom_resolution=bpoint.resolution

        plot_hic_region_matrix(
            sample,
            regionX, 
            regionY,
            resolution=resolution,
            ax=ax[i],
            minimal=True,
            show_submatrices=True,
            breakpoint_highlight=breakpoint,
        )

        xlabel_weight = "normal"
        if resolution == zoom_resolution:
            # Make spine width greater for the plot corresponding to zoom level
            ax[i].spines[["top", "right", "left", "bottom"]].set_linewidth(3)
            xlabel_weight = "bold"

        ax[i].set_xlabel(f"{int_to_resolution(resolution)} resolution\n{int_to_resolution(2*25*resolution)} window", fontweight=xlabel_weight)
        ax[i].set_ylabel("")
        
    if title is None:
        title = f"Sample={sample.id}, ZoomCenterX={chrA}:{posA}, ZoomCenterY={chrB}:{posB}"
    fig.suptitle(title, fontsize=title_fontsize, x=0.125, ha="left")

    return fig


def plot_composite_compare_two(
    sample1: ArimaPipelineSample,
    sample2: ArimaPipelineSample,
    bpoint: Breakpoint,
    figsize=(7.7, 4.3),
    resolution=50000,
    radius=3000000,
    gene_filter=None,
    title=None,
    title_fontsize=8,
    title_ha="left",
    gene_fontsize=7,
) -> plt.Figure:
    "Plot two Hi-C plots side by side (e.g. sample vs control) at a given breakfinder call."

    # Get figure and separate out axes
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    spec = GridSpec(
        ncols=5,
        nrows=2,
        figure=fig,
        height_ratios=[
            0.5,
            8,
        ],
        width_ratios=[0.5, 8, 1, 0.5, 8],
    )
    ax1_left = fig.add_subplot(spec[1, 0])
    ax1_top = fig.add_subplot(spec[0, 1])
    ax1_center = fig.add_subplot(spec[1, 1])
    divider = fig.add_subplot(spec[1, 2])
    ax2_left = fig.add_subplot(spec[1, 3])
    ax2_top = fig.add_subplot(spec[0, 4])
    ax2_center = fig.add_subplot(spec[1, 4])

    # Unpack breakfinder call
    chrA, posA = bpoint.breakendA.chrom, bpoint.breakendA.pos
    chrB, posB = bpoint.breakendB.chrom, bpoint.breakendB.pos
    
    regionX, regionY = bpoint.get_centered_regions_with_radius(radius=radius, bin_align_res=resolution)
    data1 = sample1.get_hic_region_data(regionX, regionY, resolution, measure="observed", norm_constant_normalize=True)
    data2 = sample2.get_hic_region_data(regionX, regionY, resolution, measure="observed", norm_constant_normalize=True)
    vmax = max(data1.max(), data2.max())
    if regionX.chrom == regionY.chrom:
        if regionX.overlaps(regionY):
            vmax = np.sqrt(vmax)
        else:
            vmax /= 2
        

    # Plot zoomed hic matrices in each center plot
    plot_hic_region_matrix(
        sample1,
        regionX, 
        regionY,
        resolution=resolution,
        ax=ax1_center,
        breakpoint_highlight=bpoint,
        norm_constant_normalize=True,
        vmax=vmax,
    )
    plot_hic_region_matrix(
        sample2,
        regionX, 
        regionY,
        resolution=resolution,
        ax=ax2_center,
        breakpoint_highlight=bpoint,
        norm_constant_normalize=True,
        vmax=vmax,
    )

    # Plot coverage tracks
    plot_coverage_track(sample1, regionX, resolution, ax=ax1_top, label_right=True)
    plot_coverage_track(sample1, regionY, resolution, vertical=True, ax=ax1_left)
    plot_coverage_track(sample2, regionX, resolution, ax=ax2_top, label_right=True)
    plot_coverage_track(sample2, regionY, resolution, vertical=True, ax=ax2_left )

    # Add divider
    divider.text(0.5, 0.5, "vs", ha="center", va="center", fontsize=20)
    divider.spines[["top", "right", "left", "bottom"]].set_visible(False)
    divider.xaxis.set_visible(False)
    divider.yaxis.set_visible(False)

    ax1_top.set_title(sample1.id, fontweight="bold")
    ax2_top.set_title(sample2.id + " [Control]", color="gray")
    ax1_center.text(0.05, 0.88, "(normed / 1M Hi-C reads)", transform=ax1_center.transAxes, fontsize=7, va="top", ha="left")
    ax2_center.text(0.05, 0.88, "(normed / 1M Hi-C reads)", transform=ax2_center.transAxes, fontsize=7, va="top", ha="left")

    # Make axis 2 spines a different color
    ax2_center.spines[["top", "right", "left", "bottom"]].set_color("lightgray")

    # If no specified title, then make metadata title
    if title is None:
        title = f"ZoomCenterX={chrA}:{regionX.get_center()}, ZoomCenterY={chrB}:{regionY.get_center()}\nZoomBoundsX={regionX}, ZoomBoundsY={regionY}\nZoomRes={resolution}bp, ZoomRadius={radius}bp"
    if title_ha == "left":
        fig.suptitle(title, fontsize=title_fontsize, x=0.02, ha="left")
    else:
        fig.suptitle(title, fontsize=title_fontsize)

    return fig


def plot_composite_triangle(
    sample: ArimaPipelineSample,
    bpoint: Breakpoint,
    figsize=None,
    size_base=1.5,
    vmax=None,
    measure="observed",
    norm="NONE",
    keygenes=[],
):

    fusions = bpoint.get_possible_gene_fusions()
    gene_filter = None
    fusion_style=False
    if len(fusions) > 0:
        fusion_style=True
        gene_filter = []
        for fusion in fusions:
            gene_filter.append(fusion[0].gene_name)
            gene_filter.append(fusion[1].gene_name)

    if fusion_style:
        resolution = 10000
        assembly = bpoint.get_assembly(radius=50*resolution, bin_align_res=resolution)
    else:
        resolution = 25000
        assembly = bpoint.get_assembly(radius=50*resolution, bin_align_res=resolution)
        
    assembled = sample.get_assembled_hic(assembly, resolution, norm=norm, measure=measure, gap_size=1, gap_value=0)
    
    
    width_ratios = [abs(r.bin_range[1] - r.bin_range[0]) for r in assembled.plot_regions]
    if fusion_style:
        height_ratios_tracks = [0.4, 0.3, 0.2] # Gene, Coverage, chrom arrow
    else:
        height_ratios_tracks = [0.8, 0.3, 0.2] # Gene, Coverage, chrom arrow
    
    if figsize is None:
        figheight = 2.5 + sum(height_ratios_tracks)
        figwidth = 5
        figsize = (size_base*figwidth, size_base*figheight)
    height_ratios = [2.5] + height_ratios_tracks
    fig = plt.figure(figsize=figsize)
    nrows = 1 + 3
    ncols = len(assembled.plot_regions)
    gs = fig.add_gridspec(nrows, ncols, wspace=0, hspace=0, height_ratios=height_ratios, width_ratios=width_ratios)

    ax_triangle = fig.add_subplot(gs[0, :])
    if vmax is None:
        vmax = np.nanmax(assembled.data) / 8
    plot_assembled_triangle(assembled, resolution, ax=ax_triangle, vmax=vmax, rasterized=True, aspect="auto",)
    ax_triangle.set_ylabel(sample.id, rotation=90, ha="center", va="center")



    for col, plot_region in enumerate(assembled.plot_regions):
        ax_gene = fig.add_subplot(gs[1, col])
        ax_coverage = fig.add_subplot(gs[2, col])
        ax_arrow = fig.add_subplot(gs[3, col])
        if plot_region.genomic_region is None:
            blank_axis(ax_gene)
            blank_axis(ax_coverage)
            blank_axis(ax_arrow)
            for ax in [ax_gene, ax_coverage]:
                ax.set_fc("none")
            continue
        else:
            if fusion_style and gene_filter is not None:
                plot_gene_track(plot_region.genomic_region, ax=ax_gene, all_same_line=True, gene_filter=gene_filter, min_rows=0, max_rows=1, fontsize=10)
                ax_gene.set_ylabel("Fusion", rotation=90, ha="center", va="center")
                ymin, ymax = ax_gene.get_ylim()
                ax_gene.set_ylim(ymin-0.1, ymax+0.1)
                for obj in ax_gene.findobj(matplotlib.text.Text):
                    if obj.get_text() in gene_filter:
                        obj.set_fontweight("bold")
            else:
                closest_to = bpoint.breakendA.pos if col == 0 else bpoint.breakendB.pos
                plot_gene_track(plot_region.genomic_region, ax=ax_gene, closest_to=closest_to, highlight_genes=keygenes)
                ax_gene.set_ylabel("Genes", rotation=90, ha="center", va="center")
            plot_coverage_track(sample, plot_region.genomic_region, resolution=resolution, ax=ax_coverage, label_right=True)
            plot_arrow_track(plot_region.genomic_region, ax=ax_arrow)
            for obj in ax_coverage.findobj(matplotlib.text.Text):
                if obj.get_text() == "Coverage":
                    obj.set_visible(False)
            ax_coverage.set_ylabel("Cov.", rotation=90, ha="center", va="center")
        if col > 0:
            for ax in [ax_gene,ax_arrow]:
                ax.set_ylabel("")
                ax.set_yticks([])
                blank_axis(ax)
        if col == 0:
            ax_coverage.set_yticks([])
        else:
            ax_coverage.set_ylabel("")
        for ax in [ax_gene, ax_coverage]:
            ax.set_fc("none")
        ax_gene.spines[["left", "right"]].set_visible(True)
        ax_gene.spines[["left", "right"]].set_ec("gray")
        ax_coverage.spines[["left", "right", "top"]].set_visible(True)
        ax_coverage.spines[["left", "right"]].set_ec("gray")
        ax_coverage.spines[["top"]].set_ec("lightgray")
        ax_coverage.spines[["top"]].set_linestyle(":")
        ax_coverage.spines[["top"]].set_linewidth(0.5)
        
    return fig


def plot_pseudotarget_hic_dual(
    sample: ArimaPipelineSample,
    geneA_name: str,
    geneB_name: str,
    figsize=(2,2),
    resolution=100000,
    radius=25*100000,
):
    fig, ax = plt.subplots(2, 2, figsize=figsize, width_ratios=[1, 8], height_ratios=[1, 8])
    fig.subplots_adjust(wspace=0.01, hspace=0.01)

    ax_gene_top = ax[0, 1]
    ax_gene_left = ax[1, 0]
    ax_hic = ax[1, 1]
    ax_corner = ax[0,0]
    blank_axis(ax_corner)

    try:
        geneA = GENE_ANNOTATIONS.genes_by_name(geneA_name)[0]
        geneB = GENE_ANNOTATIONS.genes_by_name(geneB_name)[0]
    except ValueError:
        blank_axis(ax_hic)
        blank_axis(ax_gene_top)
        blank_axis(ax_gene_left)
        ax_hic.text(0.5, 0.5, f"Invalid Gene Names:\n{geneA_name} & {geneB_name}", transform=ax_hic.transAxes)
        return fig

    chromA = chr_prefix(geneA.contig)
    posA = (geneA.start + geneA.end) // 2
    chromB = chr_prefix(geneB.contig)
    posB = (geneB.start + geneB.end) // 2


    regionX = GenomicRegion(chromA, posA-radius, posA+radius, bin_align_res=resolution)
    regionY = GenomicRegion(chromB, posB-radius, posB+radius, bin_align_res=resolution)

    plot_hic_region_matrix(sample, regionX, regionY, resolution, ax=ax_hic, minimal=True, show_breakpoints=False, vmax=int(sample.norm_constant))
    ax_hic.set_xlabel("")
    ax_hic.set_ylabel("")
    ax_hic.text(0.02, 0.8, "(capped)", transform=ax_hic.transAxes, fontsize=7)

    plot_gene_track(regionX, gene_filter=[geneA_name], ax=ax_gene_top, min_rows=0, all_same_line=True, fontsize=10)
    plot_gene_track(regionY, gene_filter=[geneB_name], ax=ax_gene_left, min_rows=0, all_same_line=True, fontsize=10, vertical=True)

    return fig



def plot_pseudotarget_hic_single(
    sample: ArimaPipelineSample,
    gene_name: str,
    resolution=500000,
    threshold=100,
):


    fig, ax = plt.subplots(figsize=(8, 0.25,))
    try:
        gene = GENE_ANNOTATIONS.genes_by_name(gene_name)[0]
    except ValueError:
        blank_axis(ax)
        ax.text(0.5, 0.5, f"Invalid gene name: {gene_name}", transform=ax.transAxes)
        return fig
        
    chrom = chr_prefix(gene.contig)
    width = resolution
    start = gene.start - width
    end = gene.end + width
    region = GenomicRegion(chrom, start, end)

    values = sample.get_genome_wide_virtual_4c_at_locus(region, resolution, measure="oe")
    
    # Pull all chromosomes together
    y_values = np.concatenate([ v for _, v in values ])
    # ax[i].plot(y_values, color="black")

    # Add lines for each chromosome boundary
    cumsum = 0 
    for j, (_, data) in enumerate(values):
        cumsum += len(data)
        ax.axvline(cumsum, color="#eee", lw=1)
    ax.set_yticks([])

    # Add chromosomes to x axis
    cumsum = 0 
    xlabel_pos = []
    xlabels = []
    for j, (chr_name, data) in enumerate(values):
        if chr_name == chrom:
            gene_pos = (gene.start + gene.end) / 2
            gene_pos = len(data) * gene_pos / CHROM_SIZES[chr_name]
            ax.scatter(cumsum + gene_pos, 1, color="gray", marker="+")
        rect = Rectangle((cumsum, 0), len(data), 2, fc=CHROM_COLORS[chr_name], ec="none", alpha=0.08)
        ax.add_patch(rect)
        cumsum += len(data)
        xlabel_pos.append(cumsum - len(data)//2)
        xlabels.append(chr_unprefix(chr_name))

    # Scatterplot, and if the value is above 100, color it red
    positive = False
    positive = any(v > threshold for v in y_values) > 0
    colors = ["red" if v > threshold else "black" for v in y_values]
    sizes = [20 if v > threshold else 0 for v in y_values]
    ypos = [ 1 if v > threshold else 0 for v in y_values]
    ax.scatter(range(len(y_values)), ypos, color=colors, s=sizes)
    # ax[i].scatter((anchor_gene.start+anchor_gene.end)/2, 1, color="green", s=20, alpha=0.5)
    ax.set_ylim(0, 2)
    # ax[i].imshow(np.expand_dims(y_values, axis=0), cmap=REDMAP, vmin=0, vmax=threshold*2, aspect="auto")

    ylabel_color = "crimson" if positive else "black"    
    ylabel_weight = "bold" if positive else "normal"
    ax.set_ylabel(gene_name, rotation=0, va="center", ha="right", color=ylabel_color, fontsize=14, fontweight=ylabel_weight)

    ax.set_xticks(xlabel_pos, xlabels, rotation=0, ha="center", va="top", fontsize=7)
    ax.set_xlim(0, cumsum)

    return fig, positive

