"""Plotting functions used for dashboard generation. 

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from numpy.typing import NDArray
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from hicstraw import HiCFile
from hicdash.constants import (
    CHROMS,
    CHROM_INDICES,
    CHROM_SIZES,
    GENE_ANNOTATIONS,
    BEDPE_COLORS,
    BIGWIG_COLORS,
    MARKER_SIZE_DICT,
)
from hicdash.definitions import (
    to_strand,
    QCData,
    BreakfinderCall,
    BedpeLine,
    ArimaPipelineSample,
    Pairing,
    VariantCategory,
    Strand,
    Breakpoint,
    Region,
)
from matplotlib.patches import Rectangle, Ellipse
from matplotlib.ticker import FixedLocator, MultipleLocator
from hicdash.utilities import (
    chr_prefix,
    chr_unprefix,
    to_mega,
    get_bin_extent,
    int_to_resolution,
)
import pyBigWig


# -------------------------------------------------------------------------------
# PLOTTING CONSTANTS
# -------------------------------------------------------------------------------

# Default red colormap for all Hi-C plots (similar to Juicebox) with gray masking
REDMAP = LinearSegmentedColormap.from_list("bright_red", [(1, 1, 1), (1, 0, 0)])
REDMAP.set_bad(color="gainsboro")


# -------------------------------------------------------------------------------
# PLOTTING HELPERS
# -------------------------------------------------------------------------------


def get_hic_region_data(
    sample: ArimaPipelineSample,
    regionX: Region,
    regionY: Region,
    resolution: int,
    normalization="NONE",
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

    Normalization can be "NONE", "VC", "VC_SQRT" or "KR" ("NONE" by default).
    """

    # Unpack the regions for convenience
    # chrX and chrY refer to the chromosomes on the X and Y axis, NOT chromosomes X and Y!
    chrX, startX, endX = regionX.chr, regionX.start, regionX.end
    chrY, startY, endY = regionY.chr, regionY.start, regionY.end

    # Get true extents
    startTrueX, endTrueX = get_bin_extent(startX, endX, resolution)
    startTrueY, endTrueY = get_bin_extent(startY, endY, resolution)

    # Calculate the expected size of the final hic data matrix
    # The true extents are guaranteed to align to a bin
    expX = (endTrueX - startTrueX) // resolution
    expY = (endTrueY - startTrueY) // resolution

    # Next, get the actual .hic data.
    # hicstraw only accepts first arg chr <= second arg chr, so we need to switch chromosomes and transpose if needed.
    # Also need to use the unprefixed chromosome names for hicstraw.
    # Also guard the bounds of the data retrieval to be within 0 and the chromosome sizes.
    # Retrieve one less than the end to ensure the prevent retrieving an extra bin if bin falls on bin border
    if CHROM_INDICES[chrY] <= CHROM_INDICES[chrX]:
        # Retrieves chrY on 1st axis (y axis) and chrX on 2nd axis (x axis), so no need to transpose
        zoom_data = sample.hic.getMatrixZoomData(
            chr_unprefix(chrY),
            chr_unprefix(chrX),
            "observed",
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
        zoom_data = sample.hic.getMatrixZoomData(
            chr_unprefix(chrX),
            chr_unprefix(chrY),
            "observed",
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
    data = data / sample.norm_constant

    return data


def get_hic_direct_data(
    sample: ArimaPipelineSample,
    chrX: str,
    startX: int,
    endX: int,
    chrY: str,
    startY: int,
    endY: int,
    resolution: int,
    normalization="NONE",
) -> NDArray:
    """Gets hic data direct from hicstraw (less convenient than the zoomed hic wrapper above, but useful for whole-chromosome plots)"""
    if CHROM_INDICES[chrY] <= CHROM_INDICES[chrX]:
        zoom_data = sample.hic.getMatrixZoomData(
            chr_unprefix(chrY),
            chr_unprefix(chrX),
            "observed",
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
        zoom_data = sample.hic.getMatrixZoomData(
            chr_unprefix(chrX),
            chr_unprefix(chrY),
            "observed",
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
    data = data / sample.norm_constant

    return data


# -------------------------------------------------------------------------------
# Hi-C MATRIX PLOTS
# -------------------------------------------------------------------------------


def plot_hic_region_matrix(
    sample: ArimaPipelineSample,
    regionX: Region,
    regionY: Region,
    resolution: int,
    ax: plt.Axes | None,
    minimal=False,
    show_breakfinder_calls=True,
    breakfinder_highlight:BreakfinderCall | BedpeLine | None=None,
    breakfinder_marker="+",
    breakfinder_color="black",
    normalization="NONE",
    vmax=None,
    cmap=REDMAP,
    title="",
    title_fontsize=11,
    label_fontsize=10,
    tick_fontsize=9,
    grid=False,
    crosshairs=False,
    show_submatrices=False,
    extra_bedpe: list[BedpeLine] = [],
) -> tuple[plt.Axes, tuple[int, int], tuple[int, int], list[tuple[str, int, str, float]]]:
    """Plots a specified Hi-C region.

    For most accurate alignment, the regions should be aligned at the start of a resolution bin.

    At the moment, only tested for regions of the same size (i.e. a square matrix).

    The color scale is capped at a quarter of the maximum value in the matrix by default.

    The axis limits have to be calculated here to ensure the plot is centered and axis limits are aligned with bins.

    """

    # Get plot axis (or get global axis if none provided)
    if ax is None:
        ax = plt.gca()

    # Get matrix data, then apply mask to values out of bounds (marked by -1s)
    data = get_hic_region_data(
        sample, regionX, regionY, resolution, normalization=normalization
    )
    masked = np.ma.masked_where(data < 0, data)

    # Set max of color scale to a quarter ot the max value (but at least 1)
    if vmax is None:
        vmax = max(1, masked.max() // 4)

    # Unpack region for convenience
    chrX, startX, endX = regionX.chr, regionX.start, regionX.end
    chrY, startY, endY = regionY.chr, regionY.start, regionY.end

    # Get the "true" extent of the heatmap image (only an issue if the region start and end are not resolution bins)
    # True extent here is defined as the start of the first bin to the end of the last bin
    startTrueX, endTrueX = get_bin_extent(startX, endX, resolution)
    startTrueY, endTrueY = get_bin_extent(startY, endY, resolution)
    centerTrueX = (startTrueX + endTrueX) // 2
    centerTrueY = (startTrueY + endTrueY) // 2

    ax.matshow(
        masked,
        cmap=cmap,
        vmin=0,
        vmax=vmax,
        aspect="auto",
        extent=[startTrueX, endTrueX, endTrueY, startTrueY],
    )
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Add axis labels (depending on level of detail)
    if minimal:
        # Minimal plot just has the hic matrix data and chromosome labels
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontdict={"fontsize": title_fontsize})
        ax.set_xlabel(f"{chrX}", fontdict={"fontsize": label_fontsize})
        ax.set_ylabel(f"{chrY}", fontdict={"fontsize": label_fontsize})

    else:
        # Make a start, center and end tick for each axis (and align appropriately on the axis)
        # Note: because of matshow, the bounds of the plot is actually offset by 0.5
        ax.set_xticks(
            [startTrueX, centerTrueX, endTrueX],
            map(lambda x: f"{x:,}", [startTrueX, centerTrueX, endTrueX]),
        )

        ax.set_yticks(
            [startTrueY, centerTrueY, endTrueY],
            map(lambda x: f"{x:,}", [startTrueY, centerTrueY, endTrueY]),
            rotation=90,
        )

        # Label the axes
        ax.set_xlabel(f"{chrX}", fontdict={"fontsize": label_fontsize})
        ax.set_ylabel(f"{chrY}", fontdict={"fontsize": label_fontsize}, rotation=90)
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
        xminor_ticks = FixedLocator(np.arange(startTrueX, endTrueX, resolution))
        yminor_ticks = FixedLocator(np.arange(startTrueY, endTrueY, resolution))
        ax.xaxis.set_minor_locator(xminor_ticks)
        ax.yaxis.set_minor_locator(yminor_ticks)
        ax.grid(True, which="both", linestyle="solid", linewidth=0.5, color="gainsboro")

    # Plot annotations
    plotted_crosshairs = []
    if show_breakfinder_calls and sample.breakfinder_calls is not None:

        # Iterate through annotation sets: (annotations, color)
        for annotation_set, annotation_color in list(zip(extra_bedpe, BEDPE_COLORS)) + [
            (sample.breakfinder_calls, breakfinder_color)
        ]:

            # Select only breakpoints that involve these two chromosomes
            for call in annotation_set:

                # Check if call is a breakfinder call or a generic bedpe line type
                if isinstance(call, BreakfinderCall):
                    if call.breakpointA.chr == chrX and call.breakpointB.chr == chrY:
                        posX = call.breakpointA.pos
                        posY = call.breakpointB.pos
                        callStartX = call.breakpointA.start
                        callEndX = call.breakpointA.end
                        callStartY = call.breakpointB.start
                        callEndY = call.breakpointB.end
                    elif call.breakpointA.chr == chrY and call.breakpointB.chr == chrX:
                        posX = call.breakpointB.pos
                        posY = call.breakpointA.pos
                        callStartX = call.breakpointB.start
                        callEndX = call.breakpointB.end
                        callStartY = call.breakpointA.start
                        callEndY = call.breakpointA.end
                    else:
                        continue
                elif isinstance(call, BedpeLine):
                    # Default to start coordinate if a simple bedpe line
                    if call.chrA == chrX and call.chrB == chrY:
                        posX = call.startA
                        posY = call.startB
                        callStartX = call.startA
                        callEndX = call.endA
                        callStartY = call.startB
                        callEndY = call.endB
                    elif call.chrA == chrY and call.chrB == chrX:
                        posX = call.startB
                        posY = call.startA
                        callStartX = call.startB
                        callEndX = call.endB
                        callStartY = call.startA
                        callEndY = call.endA
                    else:
                        continue

                alpha = 1
                size = MARKER_SIZE_DICT[resolution]
                mew=size / 10
                if breakfinder_highlight is not None: 
                    if call != breakfinder_highlight:
                        alpha = 0.6
                        # size *= 0.6
                        mew *= 0.6

                # If the breakpoint is within the bounds of the plot, plot it
                if startTrueX <= posX <= endTrueX and startTrueY <= posY <= endTrueY:

                    # Plot the whole submatrix of breakfinder call if start and end are different
                    # Alternatively, if a bedpe and start and end are different, then plot the rectangle
                    if (
                        (show_submatrices or isinstance(call, BedpeLine))
                        and callStartX != callEndX
                        and callStartY != callEndY
                    ):
                        # Plot rectangle
                        rect = Rectangle(
                            (callStartX, callStartY),
                            callEndX - callStartX,
                            callEndY - callStartY,
                            linewidth=1,
                            edgecolor=annotation_color,
                            facecolor="none",
                            alpha=alpha,
                        )
                        ax.add_patch(rect)


                    # Plot the marker on the plot
                    # If a simple bedpe, then only plot a marker if the start and end are the same
                    if isinstance(call, BreakfinderCall) or (
                        callStartX == callEndX and callStartY == callEndY
                    ):
                        ax.plot(
                            posX,
                            posY,
                            marker=breakfinder_marker,
                            color=annotation_color,
                            markersize=size,
                            mew=mew,
                            alpha=alpha,
                        )

                        if crosshairs:
                            ax.axvline(
                                posX,
                                color=annotation_color,
                                linestyle=(0, (1, 5)),
                                linewidth=1,
                                alpha=alpha,
                            )
                            ax.axhline(
                                posY,
                                color=annotation_color,
                                linestyle=(0, (1, 5)),
                                linewidth=1,
                                alpha=alpha,
                            )
                            plotted_crosshairs.append((chrX, posX, annotation_color, alpha))
                            plotted_crosshairs.append((chrY, posY, annotation_color, alpha))

    # Reset x and y lim, in case the plotting of the markers changed it
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    return ax, (startTrueX, endTrueX), (startTrueY, endTrueY), plotted_crosshairs


def plot_hic_centered_matrix(
    sample: ArimaPipelineSample,
    chrX: str,
    centerX: str,
    chrY: str,
    centerY: str,
    resolution: int,
    radius: int,
    **kwargs,
) -> tuple[plt.Axes, tuple[int, int], tuple[int, int], list[tuple[str, int, str, float]]]:
    """Plots a single centered Hi-C matrix and returns the axes, as well as the (startX, endX) and (startY, endY) axis limits.

    The axis limits have to be calculated here to ensure the plot is centered and axis limits are aligned with bins.

    """

    # Now starts the careful calculation of the axis limits to ensure alignment with bins.
    # Start by adjusting center and radius to nearest bin
    centerX = (centerX // resolution) * resolution
    centerY = (centerY // resolution) * resolution
    radius = (radius // resolution) * resolution

    # Next get start points (which will align exactly with start of a bin)
    startX = centerX - radius
    startY = centerY - radius

    # Next get end points, and add a bin - 1 to the end to center the plot
    endX = centerX + radius
    endY = centerY + radius

    # Make the region objects
    regionX = Region(chrX, startX, endX)
    regionY = Region(chrY, startY, endY)

    ax, (startTrueX, endTrueX), (startTrueY, endTrueY), plotted_crosshairs = plot_hic_region_matrix(
        sample, regionX, regionY, resolution, **kwargs
    )

    return ax, (startTrueX, endTrueX), (startTrueY, endTrueY), plotted_crosshairs


def plot_hic_chr_context(
    sample: ArimaPipelineSample,
    chrA: str,
    chrB: str,
    resolution: int = 2500000,
    show_breakfinder_calls=True,
    region_highlight: tuple[tuple[int, int], tuple[int, int]] | None = None,
    normalization="NONE",
    ax=None,
    cmap=REDMAP,
    tick_fontsize=10,
    extra_bedpe: list[BedpeLine] = [],
) -> plt.Axes:
    """Plots the Hi-C whole-chromosome context for a given sample.

    If chrA and chrB are the same, the plot will be a single chromosome plot.
    Otherwise, it will be a 4-box plot showing both intra-chromosomal and inter-chromosomal whole-chromosome views.

    Highlights and calls are shown on the bottom left side of the diagonal; the top right side of the diagonal is unannotated.

    """

    # Get axes
    if ax is None:
        ax = plt.gca()

    # Get HiC Data for each box
    top_left = get_hic_direct_data(
        sample,
        chrA,
        0,
        CHROM_SIZES[chrA],
        chrA,
        0,
        CHROM_SIZES[chrA],
        resolution,
        normalization=normalization,
    )
    bottom_left = get_hic_direct_data(
        sample,
        chrA,
        0,
        CHROM_SIZES[chrA],
        chrB,
        0,
        CHROM_SIZES[chrB],
        resolution,
        normalization=normalization,
    )
    top_right = get_hic_direct_data(
        sample,
        chrB,
        0,
        CHROM_SIZES[chrB],
        chrA,
        0,
        CHROM_SIZES[chrA],
        resolution,
        normalization=normalization,
    )
    bottom_right = get_hic_direct_data(
        sample,
        chrB,
        0,
        CHROM_SIZES[chrB],
        chrB,
        0,
        CHROM_SIZES[chrB],
        resolution,
        normalization=normalization,
    )

    # Stack the Hi-C data into one 4-box matrix
    top_row = np.hstack([top_left, top_right])
    bottom_row = np.hstack([bottom_left, bottom_right])
    full_matrix = np.vstack([top_row, bottom_row])

    # Calculate max colormap value (default is a fraction of the sqrt of max_value)
    max_value = np.max(full_matrix)
    vmax_large = np.sqrt(max_value) / 1.5

    # Plot the large matrix
    ax.matshow(full_matrix, cmap=cmap, vmax=vmax_large, aspect="auto")

    # Add axis lines to separate the chromosomes
    ax.axhline(top_left.shape[0] - 0.5, color="gray", linewidth=1)
    ax.axvline(top_left.shape[1] - 0.5, color="gray", linewidth=1)

    # Add chromosome tick labels
    ticks = [top_left.shape[1] // 2, top_left.shape[1] + top_right.shape[1] // 2]
    ax.set_xticks(ticks, [chrA, chrB], fontsize=tick_fontsize)
    ax.set_yticks(ticks, [chrA, chrB], fontsize=tick_fontsize, rotation=90, va="center")
    ax.xaxis.set_ticks_position("bottom")

    if show_breakfinder_calls and sample.breakfinder_calls is not None:
        # TODO: make breakfinder calls shown as submatrices

        box_width = 0.01 * full_matrix.shape[0] * 2
        box_height = 0.01 * full_matrix.shape[1] * 2
        if chrA == chrB:
            box_width /= 2
            box_height /= 2

        # Choose only calls that involve these two chromosomes
        for call in sample.breakfinder_calls:

            if isinstance(call, BreakfinderCall):
                call_chrA = call.breakpointA.chr
                call_chrB = call.breakpointB.chr
                call_posA = call.breakpointA.pos
                call_posB = call.breakpointB.pos
            elif isinstance(call, BedpeLine):
                call_chrA = call.chrA
                call_chrB = call.chrB
                call_posA = (call.startA + call.endA) // 2
                call_posB = (call.startB + call.endB) // 2
            if call_chrA == chrA and call_chrB == chrB:
                # Get position of the breakfinder call as a fraction of the chroomsome, then multiply by matrix size and offset by 0.5
                box_top = (
                    top_left.shape[0]
                    + ((call_posB / CHROM_SIZES[chrB]) * bottom_left.shape[0])
                    - 0.5
                )
                box_left = call_posA / CHROM_SIZES[chrA] * top_left.shape[1] - 0.5
            elif call_chrA == chrB and call_chrB == chrA:

                box_top = (
                    top_left.shape[0]
                    + ((call_chrA / CHROM_SIZES[chrA]) * bottom_left.shape[0])
                    - 0.5
                )
                box_left = call_chrB / CHROM_SIZES[chrB] * top_left.shape[1] - 0.5
            else:
                continue

            # Add the breakfinder call to the plot on the bottom left side of the diagonal
            ax.add_patch(
                Ellipse(
                    (box_left, box_top),
                    box_width,
                    box_height,
                    fill=False,
                    edgecolor="black",
                    linewidth=0.5,
                )
            )

    if region_highlight is not None:
        # TODO: Make region highlight the same size as actual region
        # ((region_xmin, region_xmax), (region_ymin, region_ymax)) = region_highlight
        # box_width = (region_xmax - region_xmin) / CHROM_SIZES[chrA] * bottom_left.shape[1]
        # box_height = (region_ymax - region_ymin) / CHROM_SIZES[chrB] * bottom_left.shape[0]

        box_width = 0.01 * full_matrix.shape[0] * 2
        box_height = 0.01 * full_matrix.shape[1] * 2
        if chrA == chrB:
            box_width /= 2
            box_height /= 2

        (startX, endX), (startY, endY) = region_highlight
        posA = (startX + endX) // 2
        posB = (startY + endY) // 2

        # Plot in the bottom left box
        box_top = (
            top_left.shape[0]
            + ((posB / CHROM_SIZES[chrB]) * bottom_left.shape[0])
            - 0.5
        )
        box_left = posA / CHROM_SIZES[chrA] * top_left.shape[1] - 0.5
        ax.add_patch(
            Ellipse(
                (box_left, box_top),
                box_width,
                box_height,
                fill=False,
                edgecolor="blue",
                linewidth=1,
            )
        )

    # If chrA == chrB, then show only the bottom left box (the four boxes will essentially just be all the same box, so you can just halve the axis limits)
    if chrA == chrB:
        xmin, xmax = ax.get_xlim()
        ax.set_xlim(-0.5, (xmax + 0.5) // 2 - 0.5)
        ax.set_ylim((xmax + 0.5) // 2 - 0.5, xmax)
        ax.invert_yaxis()

    return ax


def plot_full_matrix(
    sample: ArimaPipelineSample,
    ax=None,
    show_breakfinder_calls=False,
    cmap=REDMAP,
) -> plt.Axes:
    """Plot the full Hi-C matrix, with or without annotations on breakfinder calls."""

    # Collate all hic data at 2500000 resolution
    rows = []
    for chrY in CHROMS:
        row = []
        for chrX in CHROMS:
            data = get_hic_direct_data(
                sample, chrX, 0, CHROM_SIZES[chrX], chrY, 0, CHROM_SIZES[chrY], 2500000
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
    max_value = np.max(full_matrix)
    vmax = np.sqrt(max_value) / 1.5

    # Plot matrix
    ax.matshow(full_matrix, cmap=cmap, vmax=vmax)

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

    if show_breakfinder_calls and sample.breakfinder_calls is not None:
        for call in sample.breakfinder_calls:
            if isinstance(call, BreakfinderCall):
                pairing = call.pairing
                chrY = call.breakpointA.chr
                chrX = call.breakpointB.chr
                posA = call.breakpointA.pos
                posB = call.breakpointB.pos
            elif isinstance(call, BedpeLine):
                if call.chrA == call.chrB:
                    pairing = Pairing.INTRA
                else:
                    pairing = Pairing.INTER
                chrY = call.chrA
                chrX = call.chrB
                posA = (call.startA + call.endA) // 2
                posB = (call.startB + call.endB) // 2
            # Show only inter-chromosomal calls for now
            if pairing == Pairing.INTER:

                # Calculate position on full-matrix as a fraction of the chromosome
                idxA = CHROM_INDICES[chrY]
                pctA = posA / CHROM_SIZES[chrY]
                coordA = sum(new_sizes[:idxA]) + (pctA * new_sizes[idxA])

                idxB = CHROM_INDICES[chrX]
                pctB = posB / CHROM_SIZES[chrX]
                coordB = sum(new_sizes[:idxB]) + (pctB * new_sizes[idxB])

                radius = 0.02 * full_matrix.shape[1]
                ellipse1 = Ellipse(
                    (coordB, coordA),
                    radius,
                    radius,
                    fill=False,
                    edgecolor="blue",
                    linewidth=1,
                )
                ellipse2 = Ellipse(
                    (coordA, coordB),
                    radius,
                    radius,
                    fill=False,
                    edgecolor="blue",
                    linewidth=1,
                )
                ax.add_patch(ellipse1)
                ax.add_patch(ellipse2)

    return ax


# -------------------------------------------------------------------------------
# TRACKS
# -------------------------------------------------------------------------------


def plot_gene_track(
    chr: str,
    start: int,
    end: int,
    gene_filter: list[str] | None = None,
    hide_axes=True,
    vertical=False,
    ax=None,
    fontsize=8,
    max_rows=6,
    min_rows=3,
    protein_coding_only=True,
    crosshairs: bool=False, 
    plotted_crosshairs: list[tuple[str, int, str, float]]=[],
    centered_names=False,
) -> plt.Axes:
    """Plot a gene track (based on GENE_ANNOTATIONS) for a given range.

    Note: if too many genes are in the given range, then only a subset of genes will be plotted.
    (otherwise the track will be too crowded to read).
    """

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
                    lambda g: g.biotype == "protein_coding" and g.gene_name != "",
                    candidates,
                )
            )
        else:
            genes = list(filter(lambda g: g.gene_name != "", candidates))

        # Narrow down the genes if there are too many to plot, keeping only a number of genes around the center
        # Keep genes whose midpoints are closest to the center
        genes.sort(key=lambda g: abs(center - (g.start + g.end) / 2))
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
        ax.yaxis.set_visible(False)
        ax.xaxis.set_visible(False)
        ax.spines[["top", "right", "left", "bottom"]].set_visible(False)

    # Calculate values for plotting

    # Keep track of how many genes are plotted
    plot_counter = 0

    # Use 5 percent as a temporary padding value
    pct_5 = 5 * (end - start) / 100
    max_arrowhead_width = (end - start) / 75
    plot_line_width = 0.4

    # Get genes which intersect the center directly
    direct_genes = GENE_ANNOTATIONS.genes_at_locus(
        contig=chr_unprefix(chr), position=center
    )
    direct_genes_set = set([gene.gene_name for gene in direct_genes])

    for gene in genes:

        # Plot base arrow indicating strandness
        arr_start, arr_end = (
            (gene.start, gene.end - gene.start)
            if gene.strand == "+"
            else (gene.end, gene.start - gene.end)
        )
        if vertical:
            ax.arrow(
                plot_counter,
                arr_start,
                0,
                arr_end,
                head_width=plot_line_width,
                head_length=max_arrowhead_width,
                length_includes_head=False,
                fc="white",
            )
        else:
            ax.arrow(
                arr_start,
                plot_counter,
                arr_end,
                0,
                head_width=plot_line_width,
                head_length=max_arrowhead_width,
                length_includes_head=False,
                fc="white",
            )

        # Plot each exon as a rectangle on the gene line
        for exon in gene.exons:
            exon_length = exon.end - exon.start
            exon_width_start = plot_counter - (plot_line_width / 2)
            exon_width = plot_line_width
            if vertical:
                ax.add_patch(
                    Rectangle(
                        (exon_width_start, exon.start),
                        exon_width,
                        exon_length,
                        edgecolor="black",
                        facecolor="dodgerblue",
                    )
                )
            else:
                ax.add_patch(
                    Rectangle(
                        (exon.start, exon_width_start),
                        exon_length,
                        exon_width,
                        edgecolor="black",
                        facecolor="dodgerblue",
                    )
                )

        # Get text position - and avoid putting the text out of axes
        # Bolden the text if passes directly through the center of the range (assuming a breakpoint is in the center)
        # fontweight = "bold" if gene.gene_name in direct_genes_set else "normal"
        fontweight = "normal"
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
                ax.text(
                    plot_counter+plot_line_width/2,
                    text_position,
                    gene.gene_name,
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
                text_position = gene.start - (pct_5 * 0.25)
                if gene.strand == "-":
                    text_position -= max_arrowhead_width
                if text_position - pct_5 < ax.get_ylim()[1]:
                    text_position = gene.end + (pct_5 * 0.25)
                    va = "top"
                    if gene.strand == "+":
                        text_position += max_arrowhead_width
                    if text_position + pct_5 > ax.get_ylim()[0]:
                        text_position = ax.get_ylim()[0]
                        va = "bottom"
                        row_position = plot_counter + plot_line_width / 2
                ax.text(
                    row_position,
                    text_position,
                    gene.gene_name,
                    ha="center",
                    va=va,
                    fontsize=fontsize,
                    rotation=90,
                    color=color,
                    fontweight=fontweight,
                ).set_clip_on(True)
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
                ax.text(
                    text_position,
                    plot_counter+plot_line_width/2,
                    gene.gene_name,
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
                if gene.strand == "-":
                    text_position -= max_arrowhead_width
                if text_position - pct_5 < ax.get_xlim()[0]:
                    text_position = gene.end + (pct_5 * 0.25)
                    ha = "left"
                    if gene.strand == "+":
                        text_position += max_arrowhead_width
                    if text_position + pct_5 > ax.get_xlim()[1]:
                        text_position = ax.get_xlim()[1]
                        ha = "right"
                        row_position = plot_counter + plot_line_width / 2
                ax.text(
                    text_position,
                    row_position,
                    gene.gene_name,
                    ha=ha,
                    va="center",
                    fontsize=fontsize,
                    color=color,
                    fontweight=fontweight,
                ).set_clip_on(True)

        # Increment plot counter
        plot_counter += 1

    # Extend the axis limits if centered_gene_names is used
    if centered_names and plot_counter >= min_rows:
        if vertical:
            xmax, xmin = ax.get_xlim()
            ax.set_xlim(xmax + plot_line_width / 2, xmin)
        else:
            ymin, ymax = ax.get_ylim()
            ax.set_ylim(ymin, ymax + plot_line_width / 2)

    # If only one gene was plotted, then center it 
    if plot_counter == 1:
        if vertical:
            xmax, xmin = ax.get_xlim()
            ax.set_xlim(xmax + plot_line_width / 2, xmin - plot_line_width / 2)
        else:
            ymin, ymax = ax.get_ylim()
            ax.set_ylim(ymin - plot_line_width / 2, ymax + plot_line_width / 2)

    # If only a few genes were plotted, then add a bit of space padding to the plot
    elif plot_counter < min_rows:
        if vertical:
            ax.set_xlim(min_rows + plot_line_width, 0 - plot_line_width)
        else:
            ax.set_ylim(0 - plot_line_width, min_rows + plot_line_width)

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

    return ax


def plot_coverage_track(
    sample: ArimaPipelineSample,
    chr: str,
    start: int,
    end: int,
    resolution: int,
    max_coverage=5,
    ax=None,
    hide_axes=True,
    vertical=False,
    fontsize=8,
    bar_color="#61B8D1",
    label="Coverage",
    label_fontsize=8,
    crosshairs: bool=False, 
    plotted_crosshairs: list[tuple[str, int, str]]=[],
) -> plt.Axes:
    """Plot a coverage track for a given chromosome region.

    For best results, start and end should be multiples of the resolution.

    """

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
    else:
        ax.bar(positions, norm_vector, width=resolution, color=bar_color)
        ax.set_xlim(start, end)
        ax.set_ylim(0, max_coverage)

    if hide_axes:
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
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


def plot_bigwig_track(
    bw_handle: object,
    chr: str,
    start: int,
    end: int,
    num_bins=1000,
    hide_axes=True,
    vertical=False,
    ax=None,
    fontsize=8,
    label: str = "",
    color="blue",
    crosshairs: bool=False, 
    plotted_crosshairs: list[tuple[str, int, str, float]]=[],
) -> plt.Axes:

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

    # TODO: Using an arbitrary normalization for now, but probably want to choose a different normalization at some point
    normalizer = np.sqrt(bw_handle.header()["maxVal"])

    positions = np.linspace(start, end, num_bins)

    # Plot the data depending on if horizontal or vertical axis
    if ax is None:
        ax = plt.gca()

    if vertical:
        # Fill from right to data peak on left
        ax.fill_betweenx(positions, np.zeros(num_bins), data, color=color)
        ax.set_ylim(start, end)
        ax.invert_yaxis()
        ax.set_xlim(0, normalizer)
        ax.invert_xaxis()
    else:
        ax.fill_between(positions, np.zeros(num_bins), data, color=color)
        ax.set_xlim(start, end)
        ax.set_ylim(0, normalizer)

    if hide_axes:
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
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



# -------------------------------------------------------------------------------
# USEFUL COMPOSITE PLOTS
# -------------------------------------------------------------------------------


def plot_composite_double_whole_matrix(
    sample: ArimaPipelineSample, figsize=(13.5, 7), title_fontsize=14, **kwargs
) -> plt.Figure:
    """Plot sample whole matrix, unannotated next to annotated with translocations"""
    fig, ax = plt.subplots(1, 2, figsize=figsize)
    plot_full_matrix(sample, ax=ax[0], **kwargs)
    plot_full_matrix(sample, ax=ax[1], show_breakfinder_calls=True, **kwargs)
    plt.suptitle(sample.id + "\n", fontsize=title_fontsize)
    plt.tight_layout()
    return fig


def plot_composite_context_and_zoom(
    sample: ArimaPipelineSample,
    call: BreakfinderCall,
    figsize=(13, 7.3),
    zoom_resolution=10000,
    zoom_radius=200000,
    gene_filter=None,
    title=None,
    title_fontsize=8,
    title_ha="left",
    gene_fontsize=7,
    extra_bedpe: list[BedpeLine] = [],
    coverage_track=True,
    hide_track_axes=True,
    extra_bigwig_handles: list[tuple[str, object]] = [],
    crosshairs=False,
    grid=False,
    plot_at_call_resolution=False,
    min_gene_rows=3,
    centered_gene_names=False,
    **kwargs,
) -> plt.Figure:
    """Plot whole-chromosome context on left and zoomed breakfinder call on right with gene track."""

    # Get figure and separate out axes
    fig = plt.figure(figsize=figsize, constrained_layout=True)

    # Calculate the number of rows and columns
    num_bigwig_tracks = len(extra_bigwig_handles)
    sizes_bigwig = [0.5] * num_bigwig_tracks
    num_coverage_tracks = 1 if coverage_track else 0
    sizes_coverage = [0.5] * num_coverage_tracks

    num_rows = 2 + num_coverage_tracks + num_bigwig_tracks
    num_cols = 3 + num_coverage_tracks + num_bigwig_tracks
    height_ratios = sizes_coverage + sizes_bigwig + [1, 8]
    width_ratios = [sum(height_ratios)] + sizes_coverage + sizes_bigwig + [1, 8]

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

    # Unpack breakfinder call
    if isinstance(call, BreakfinderCall):
        chrA, posA = call.breakpointA.chr, call.breakpointA.pos
        chrB, posB = call.breakpointB.chr, call.breakpointB.pos
    else:
        chrA, posA = call.chrA, (call.startA + call.endA) // 2
        chrB, posB = call.chrB, (call.startB + call.endB) // 2

    # Plot zoomed hic matrix first to get axis bounds
    if plot_at_call_resolution and isinstance(call, BreakfinderCall):
        zoom_resolution = call.resolution
        zoom_radius = 20 * zoom_resolution

    _, (xmin, xmax), (ymin, ymax), plotted_crosshairs = plot_hic_centered_matrix(
        sample,
        chrA,
        posA,
        chrB,
        posB,
        resolution=zoom_resolution,
        radius=zoom_radius,
        ax=ax_zoom,
        extra_bedpe=extra_bedpe,
        crosshairs=crosshairs,
        grid=grid,
        breakfinder_highlight=call,
        **kwargs,
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

    _ = plot_hic_chr_context(
        sample,
        chrA,
        chrB,
        context_resolution,
        show_breakfinder_calls=True,
        region_highlight=((xmin, xmax), (ymin, ymax)),
        ax=ax_large,
        extra_bedpe=extra_bedpe,
    )

    # Plot coverage tracks
    if coverage_track:
        ax_coverage_top = fig.add_subplot(spec[0, ax_zoom_col])
        ax_coverage_bottom = fig.add_subplot(spec[ax_zoom_row, 1])
        plot_coverage_track(
            sample,
            chrA,
            xmin,
            xmax,
            zoom_resolution,
            ax=ax_coverage_top,
            hide_axes=hide_track_axes,
            label_fontsize=7,
            crosshairs=crosshairs, 
            plotted_crosshairs=plotted_crosshairs,
        )
        plot_coverage_track(
            sample,
            chrB,
            ymin,
            ymax,
            zoom_resolution,
            vertical=True,
            ax=ax_coverage_bottom,
            hide_axes=hide_track_axes,
            label_fontsize=7,
            crosshairs=crosshairs, 
            plotted_crosshairs=plotted_crosshairs,
        )

    # Plot gene tracks
    plot_gene_track(
        chrA,
        xmin,
        xmax,
        ax=ax_genes_top,
        fontsize=gene_fontsize,
        gene_filter=gene_filter,
        hide_axes=hide_track_axes,
        crosshairs=crosshairs, 
        plotted_crosshairs=plotted_crosshairs,
        min_rows=min_gene_rows,
        centered_names=centered_gene_names,
    )
    plot_gene_track(
        chrB,
        ymin,
        ymax,
        ax=ax_genes_left,
        vertical=True,
        fontsize=gene_fontsize,
        gene_filter=gene_filter,
        hide_axes=hide_track_axes,
        crosshairs=crosshairs, 
        plotted_crosshairs=plotted_crosshairs,
        min_rows=min_gene_rows,
        centered_names=centered_gene_names,
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
            chrA,
            xmin,
            xmax,
            label=label,
            ax=ax_horizontal,
            hide_axes=hide_track_axes,
            color=color,
            fontsize=7,
            crosshairs=crosshairs, 
            plotted_crosshairs=plotted_crosshairs,
        )
        plot_bigwig_track(
            bw_handle,
            chrB,
            ymin,
            ymax,
            label=label,
            ax=ax_vertical,
            vertical=True,
            hide_axes=hide_track_axes,
            color=color,
            fontsize=7,
            crosshairs=crosshairs, 
            plotted_crosshairs=plotted_crosshairs,
        )

    # If no specified title, then make metadata title
    if title is None:
        title = f"Sample={sample.id}\nZoomCenterX={chrA}:{posA}, ZoomCenterY={chrB}:{posB}\nZoomBoundsX={chrA}:{xmin}-{xmax}, ZoomBoundsY={chrB}:{ymin}-{ymax}\nZoomRes={zoom_resolution}bp, ZoomRadius={zoom_radius}bp"
    if title_ha == "left":
        fig.suptitle(title, fontsize=title_fontsize, x=0.02, ha="left")
    else:
        fig.suptitle(title, fontsize=title_fontsize)

    return fig

def plot_composite_multires_breakpoint(
    sample: ArimaPipelineSample,
    call: BreakfinderCall | BedpeLine,
    extra_bedpe: list[BedpeLine] = [],
    figheight=2.5,
    default_zoom_resolution=10000,
) -> plt.Figure: 
    """Plot a single breakpoint at multiple resolutions in a row."""

    resolutions = [1000000, 500000, 100000, 50000, 10000, 5000, 1000]
    num_plots = len(resolutions)
    fig, ax = plt.subplots(1, num_plots, figsize=(num_plots * figheight, figheight))

    for i, resolution in enumerate(resolutions):
        if isinstance(call, BreakfinderCall):
            chrA, posA = call.breakpointA.chr, call.breakpointA.pos
            chrB, posB = call.breakpointB.chr, call.breakpointB.pos
            zoom_resolution=call.resolution
        else:
            chrA, posA = call.chrA, (call.startA + call.endA) // 2
            chrB, posB = call.chrB, (call.startB + call.endB) // 2
            zoom_resolution=default_zoom_resolution

        _, _, _, _ = plot_hic_centered_matrix(
            sample,
            chrA,
            posA,
            chrB,
            posB,
            resolution=resolution,
            radius=20*resolution,
            ax=ax[i],
            minimal=True,
            show_submatrices=True,
            extra_bedpe=extra_bedpe,
            breakfinder_highlight=call,
        )

        xlabel_weight = "normal"
        if resolution == zoom_resolution:
            # Make spine width greater for the plot corresponding to zoom level
            ax[i].spines[["top", "right", "left", "bottom"]].set_linewidth(3)
            xlabel_weight = "bold"

        ax[i].set_xlabel(f"{int_to_resolution(resolution)} resolution\n{int_to_resolution(2*20*resolution)} window", fontweight=xlabel_weight)
        ax[i].set_ylabel("")

    return fig


def plot_composite_compare_two(
    sample1: ArimaPipelineSample,
    sample2: ArimaPipelineSample,
    call: BreakfinderCall | BedpeLine,
    figsize=(13.5, 7.3),
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
    if isinstance(call, BreakfinderCall):
        chrA, posA = call.breakpointA.chr, call.breakpointA.pos
        chrB, posB = call.breakpointB.chr, call.breakpointB.pos
    elif isinstance(call, BedpeLine):
        chrA, posA = call.chrA, (call.startA + call.endA) // 2
        chrB, posB = call.chrB, (call.startB + call.endB) // 2

    # Plot zoomed hic matrices in each center plot
    _, (xmin1, xmax1), (ymin1, ymax1), _ = plot_hic_centered_matrix(
        sample1,
        chrA,
        posA,
        chrB,
        posB,
        resolution=resolution,
        radius=radius,
        ax=ax1_center,
        breakfinder_highlight=call,
    )
    _, (xmin2, xmax2), (ymin2, ymax2), _ = plot_hic_centered_matrix(
        sample2,
        chrA,
        posA,
        chrB,
        posB,
        resolution=resolution,
        radius=radius,
        ax=ax2_center,
        breakfinder_highlight=call,
    )

    # Assert axis limits are the same
    assert xmin1 == xmin2
    assert xmax1 == xmax2
    assert ymin1 == ymin2
    assert ymax1 == ymax2

    # Plot coverage tracks
    plot_coverage_track(sample1, chrA, xmin1, xmax1, resolution, ax=ax1_top)
    plot_coverage_track(
        sample1, chrB, ymin1, ymax1, resolution, vertical=True, ax=ax1_left
    )
    plot_coverage_track(sample2, chrA, xmin2, xmax2, resolution, ax=ax2_top)
    plot_coverage_track(
        sample2, chrB, ymin2, ymax2, resolution, vertical=True, ax=ax2_left
    )

    # Add divider
    divider.text(0.5, 0.5, "vs", ha="center", va="center", fontsize=20)
    divider.spines[["top", "right", "left", "bottom"]].set_visible(False)
    divider.xaxis.set_visible(False)
    divider.yaxis.set_visible(False)

    ax1_top.set_title(sample1.id + "\n")
    ax2_top.set_title(sample2.id + " [Control]", color="gray")

    # Make axis 2 spines a different color
    ax2_center.spines[["top", "right", "left", "bottom"]].set_color("lightgray")

    # If no specified title, then make metadata title
    if title is None:
        title = f"ZoomCenterX={chrA}:{posA}, ZoomCenterY={chrB}:{posB}\nZoomBoundsX={chrA}:{xmin1}-{xmax1}, ZoomBoundsY={chrB}:{ymin1}-{ymax1}\nZoomRes={resolution}bp, ZoomRadius={radius}bp"
    if title_ha == "left":
        fig.suptitle(title, fontsize=title_fontsize, x=0.02, ha="left")
    else:
        fig.suptitle(title, fontsize=title_fontsize)

    return fig


# -------------------------------------------------------------------------------
# QC Plot
# -------------------------------------------------------------------------------


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
    plt.xlim(0, 600)
    plt.ylim(0.5, 4)
    plt.text(0, 2, "Total informative pairs (million pairs)", fontsize=12, va="bottom")
    plt.gca().spines[["top", "left", "right"]].set_visible(False)
    plt.text(raw_pairs + 3, 1, f"{int(raw_pairs)} million raw pairs", color="black", ha="left", va="center", fontsize=9)
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
    plt.xlim([0, 1200])
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
    num_sv_calls = len(sample.breakfinder_calls)
    plt.barh(0, num_sv_calls, color="dimgray")
    plt.xlim([0, 100])
    plt.yticks([])
    plt.ylim(-0.5, 1)
    plt.text(0, 0.5, f"Breakpoints\n{num_sv_calls}", fontsize=11, va="bottom")
    plt.gca().spines[['top', 'left', 'right']].set_visible(False)

    
    return fig
