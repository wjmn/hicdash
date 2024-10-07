
import unittest
from io import StringIO

from hicdash.utilities import *
from hicdash.constants import *
from hicdash.definitions import *

class TestGenomicPosition(unittest.TestCase):

    def test_init_prefixed(self):
        gpos = GenomicPosition("chr3", 25)
        self.assertEqual(gpos.chrom, "chr3")
        self.assertEqual(gpos.pos, 25)

    def test_init_unprefixed(self):
        gpos = GenomicPosition("3", 25)
        self.assertEqual(gpos.chrom, "chr3")
        self.assertEqual(gpos.pos, 25)
        
    def test_init_float(self):
        gpos = GenomicPosition("3", 3948.0)
        self.assertEqual(gpos.chrom, "chr3")
        self.assertEqual(gpos.pos, 3948)

    def test_get_genes_unstranded(self):
        gpos = GenomicPosition("chr8", 127739000)
        genes = gpos.get_intersecting_genes()
        self.assertEqual(len(genes), 1)
        self.assertEqual(genes[0].gene_name, "MYC")
        
    def test_get_genes_stranded_pos(self):
        gpos = GenomicPosition("chr8", 127739000)
        genes = gpos.get_intersecting_genes(strand=Strand.POS)
        self.assertEqual(len(genes), 1)
        self.assertEqual(genes[0].gene_name, "MYC")
        
    def test_get_genes_stranded_neg(self):
        gpos = GenomicPosition("chr8", 127700000)
        genes = gpos.get_intersecting_genes(strand=Strand.NEG)
        self.assertEqual(len(genes), 1)
        self.assertEqual(genes[0].gene_name, "CASC11")

    def test_lt_same_chrom(self):
        gpos_less = GenomicPosition("chr8", 127700000)
        gpos_more = GenomicPosition("chr8", 127800000)
        self.assertTrue(gpos_less < gpos_more)
        
    def test_lt_diff_chrom_true(self):
        gpos_less = GenomicPosition("chr3", 127700000)
        gpos_more = GenomicPosition("chr11", 2930)
        self.assertTrue(gpos_less < gpos_more)
        
    def test_lt_diff_chrom_false(self):
        gpos_less = GenomicPosition("chr13", 10)
        gpos_more = GenomicPosition("chr5", 2930)
        self.assertFalse(gpos_less < gpos_more)
        
    def test_le_equal(self):
        gpos_less = GenomicPosition("chr3", 4)
        gpos_more = GenomicPosition(3, 4)
        self.assertTrue(gpos_less <= gpos_more)
        
    def test_le_diff_chrom_true(self):
        gpos_less = GenomicPosition("chr3", 127700000)
        gpos_more = GenomicPosition("chr11", 2930)
        self.assertTrue(gpos_less <= gpos_more)
        
    def test_le_diff_chrom_false(self):
        gpos_less = GenomicPosition("chr13", 10)
        gpos_more = GenomicPosition("chr5", 2930)
        self.assertFalse(gpos_less <= gpos_more)

    def test_get_region_with_symmetrical_radius(self):
        gpos = GenomicPosition("chr8", 127750000)
        gregion = gpos.get_region_with_symmetrical_radius(1000000)
        self.assertEqual(gregion.chrom, "chr8")
        self.assertEqual(gregion.start, 126750000)
        self.assertEqual(gregion.end, 128750000)
        
    def test_get_region_with_symmetrical_radius_bin_aligned(self):
        gpos = GenomicPosition("chr8", 127123456)
        gregion = gpos.get_region_with_symmetrical_radius(1000000, bin_align_res=5000)
        self.assertEqual(gregion.chrom, "chr8")
        self.assertEqual(gregion.start, 126120000)
        self.assertEqual(gregion.end, 128120000)

class TestGenomicRegion(unittest.TestCase):


    def test_init_unaligned(self):
        gregion = GenomicRegion("chr3", 123456, 567890)
        self.assertEqual(gregion.chrom, "chr3")
        self.assertEqual(gregion.start, 123456)
        self.assertEqual(gregion.end, 567890)
        self.assertEqual(gregion.bin_align_res, None)
        self.assertEqual(gregion.reverse, False)

    def test_init_unaligned_unprefixed(self):
        gregion = GenomicRegion("3", 123456, 567890)
        self.assertEqual(gregion.chrom, "chr3")
        self.assertEqual(gregion.start, 123456)
        self.assertEqual(gregion.end, 567890)

    def test_init_align(self):
        gregion = GenomicRegion("chr8", 126894839, 128498283, bin_align_res=25000)
        self.assertEqual(gregion.chrom, "chr8")
        self.assertEqual(gregion.start, 126875000)
        self.assertEqual(gregion.end, 128500000)
        self.assertEqual(gregion.bin_align_res, 25000)
        
    def test_init_align_no_action_needed(self):
        gregion = GenomicRegion("chr8", 127010000, 128055000, bin_align_res=5000)
        self.assertEqual(gregion.chrom, "chr8")
        self.assertEqual(gregion.start, 127010000)
        self.assertEqual(gregion.end, 128055000)
        self.assertEqual(gregion.bin_align_res, 5000)

    def test_get_bin_aligned(self):
        gregion_unaligned = GenomicRegion("chr8", 126894839, 128498283, reverse=True)
        gregion_aligned = gregion_unaligned.get_bin_aligned(resolution=25000)
        self.assertEqual(gregion_aligned.chrom, "chr8")
        self.assertEqual(gregion_aligned.start, 126875000)
        self.assertEqual(gregion_aligned.end, 128500000)
        self.assertEqual(gregion_aligned.reverse, True)
        self.assertEqual(gregion_aligned.bin_align_res, 25000)

    def test_get_reverse(self):
        gregion_original = GenomicRegion("chr8", 126894839, 128498283, bin_align_res=25000)
        gregion = gregion_original.get_reverse_true()
        self.assertEqual(gregion.chrom, "chr8")
        self.assertEqual(gregion.start, 126875000)
        self.assertEqual(gregion.end, 128500000)
        self.assertEqual(gregion.bin_align_res, 25000)
        self.assertEqual(gregion.reverse, True)

    def test_contained_genes(self):
        gregion = GenomicRegion("chr8", 127700000, 128000000)
        genes = gregion.get_contained_genes()
        self.assertEqual(len(genes), 8)
        
    def test_contained_genes_stranded(self):
        gregion = GenomicRegion("chr8", 127700000, 128000000)
        genes = gregion.get_contained_genes(strand=Strand.POS)
        self.assertEqual(len(genes), 6)
        self.assertEqual(len([g for g in genes if is_protein_coding(g)]), 1)


class TestBreakend(unittest.TestCase):
    def test_init_breakend(self):
        breakend = Breakend("chr8", 127700000, "-")
        self.assertEqual(breakend.chrom, "chr8")
        self.assertEqual(breakend.pos, 127700000)
        self.assertEqual(breakend.strand, Strand.NEG)

    def test_get_region_with_stranded_radius(self):
        breakend = Breakend(8, 127700000, "-")
        gregion = breakend.get_region_with_stranded_radius(radius=1000000)
        self.assertEqual(gregion.chrom, "chr8")
        self.assertEqual(gregion.start, 127700000)
        self.assertEqual(gregion.end, 128700000)
        self.assertEqual(gregion.reverse, False)
        
    def test_get_region_with_stranded_radius_bin_align(self):
        breakend = Breakend(8, 127123456, "+")
        gregion = breakend.get_region_with_stranded_radius(radius=1000000, bin_align_res=5000)
        self.assertEqual(gregion.chrom, "chr8")
        self.assertEqual(gregion.start, 126125000)
        self.assertEqual(gregion.end, 127125000)
        self.assertEqual(gregion.reverse, False)

class TestBreakpoint(unittest.TestCase):

    def test_init_breakpoint_sort_inter(self):
        bpoint = Breakpoint("chr14", 104400000, "+", "chr8", 127700000, "-")
        self.assertEqual(bpoint.breakendA.chrom, "chr8")
        self.assertEqual(bpoint.breakendA.pos, 127700000)
        self.assertEqual(bpoint.breakendB.chrom, "chr14")
        self.assertEqual(bpoint.breakendB.pos, 104400000)
        self.assertEqual(bpoint.resolution, None)

    def test_init_breakpoint_sort_intra(self):
        bpoint = Breakpoint("chr8", 104400000, "+", "chr8", 5000, "-")
        self.assertEqual(bpoint.breakendA.chrom, "chr8")
        self.assertEqual(bpoint.breakendA.pos, 5000)
        self.assertEqual(bpoint.breakendB.chrom, "chr8")
        self.assertEqual(bpoint.breakendB.pos, 104400000)
        self.assertEqual(bpoint.resolution, None)
        
    def test_init_breakpoint_already_sorted(self):
        bpoint = Breakpoint("chr8", 104400000, "+", "chr8", 105600000, "-")
        self.assertEqual(bpoint.breakendA.chrom, "chr8")
        self.assertEqual(bpoint.breakendA.pos, 104400000)
        self.assertEqual(bpoint.breakendB.chrom, "chr8")
        self.assertEqual(bpoint.breakendB.pos, 105600000)
        self.assertEqual(bpoint.resolution, None)

    def test_get_reconstructed_regions_with_radius(self):
        bpoint = Breakpoint("chr14", 104987654, "+", "chr8", 127123456, "-")
        region_left, region_right = bpoint.get_reconstructed_regions_with_radius(radius=1000000, bin_align_res=5000)
        self.assertEqual(region_left.chrom, "chr8")
        self.assertEqual(region_left.start, 127120000)
        self.assertEqual(region_left.end, 128120000)
        self.assertEqual(region_left.reverse, True)
        self.assertEqual(region_left.bin_align_res, 5000)
        self.assertEqual(region_right.chrom, "chr14")
        self.assertEqual(region_right.start, 103990000)
        self.assertEqual(region_right.end, 104990000)
        self.assertEqual(region_right.reverse, True)

    def test_get_centered_regions_with_radius(self):
        bpoint = Breakpoint("chr14", 104987654, "+", "chr8", 127123456, "-")
        regionA, regionB = bpoint.get_centered_regions_with_radius(radius=1000000, bin_align_res=5000)
        self.assertEqual(regionA.chrom, "chr8")
        self.assertEqual(regionA.start, 126120000)
        self.assertEqual(regionA.end, 128120000)
        self.assertEqual(regionA.reverse, False)
        self.assertEqual(regionA.bin_align_res, 5000)
        self.assertEqual(regionB.chrom, "chr14")
        self.assertEqual(regionB.start, 103990000)
        self.assertEqual(regionB.end, 105990000)
        self.assertEqual(regionB.reverse, False)

    def test_get_strandness_type_trans(self):
        bpoint = Breakpoint("chr14", 104987654, "+", "chr8", 127123456, "-")
        self.assertEqual(bpoint.get_strandness_type(), "translocation")
        
    def test_get_strandness_type_dup(self):
        bpoint = Breakpoint("chr8", 123456789, "+", "chr8", 456789, "-")
        self.assertEqual(bpoint.get_strandness_type(), "duplication")

    def test_get_deleted_region(self):
        bpoint = Breakpoint("chr8", 123456, "+", "chr8", 456789, "-")
        gregion = bpoint.get_deleted_region()
        self.assertEqual(bpoint.breakendA.chrom, bpoint.breakendB.chrom)
        self.assertEqual(bpoint.get_strandness_type(), "deletion")
        self.assertEqual(gregion.start, 123456)
        self.assertEqual(gregion.end, 456789)

    def test_lt_false_same_chrom(self):
        bpointA = Breakpoint("chr8", 123456, "+", "chr8", 456789, "-")
        bpointB = Breakpoint("chr8", 0, "+", "chr8", 456789, "-")
        self.assertEqual(bpointA < bpointB, False)
        
    def test_lt_true_diff_chrom(self):
        bpointA = Breakpoint("chr2", 900000, "+", "chr8", 456789, "-")
        bpointB = Breakpoint("chr8", 0, "+", "chr8", 456789, "-")
        self.assertEqual(bpointA < bpointB, True)

    def test_le_true_equal(self):
        bpointA = Breakpoint("chr2", 900000, "+", "chr8", 456789, "-")
        bpointB = Breakpoint("chr2", 900000, "+", "chr8", 456789, "-")
        self.assertEqual(bpointA <= bpointB, True)

    def test_from_breakfinder_file(self):
        file = StringIO("""#chr1	x1	x2	chr2	y1	y2	strand1	strand2	resolution	-logP
chr12	22566000	22623000	chr21	24135000	24151000	-	-	1kb	79.2279
chr5	51012000	51100000	chr6	37813000	37850000	+	-	1kb	285.252
chr9	130731000	131050000	chr22	22920000	23290000	-	+	1kb	69612
chr9	131030000	131199000	chr22	16429000	16820000	+	+	1kb	3967
chr9	131130000	131280000	chr13	107850000	108009000	+	+	1kb	9479.12""")
        breakpoints = Breakpoint.list_from_breakfinder_file(file)
        self.assertEqual(len(breakpoints), 5)
        first = breakpoints[0]
        self.assertEqual(first.breakendA.chrom, "chr5")
        self.assertEqual(first.breakendB.chrom, "chr6")
        self.assertEqual(first.breakendA.strand, Strand.POS)
        self.assertEqual(first.breakendB.strand, Strand.NEG)
        self.assertEqual(first.resolution, 1000)
        