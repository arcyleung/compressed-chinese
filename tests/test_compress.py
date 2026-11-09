"""Placement search shares broad strokes and keeps each glyph recognizable."""

import unittest
from pathlib import Path

import numpy as np
from PIL import ImageFont

import compress

ROOT = Path(__file__).resolve().parents[1]
FONT = ROOT / "ZenOldMincho-Regular.ttf"


def _glyph(char: str, mask: np.ndarray) -> compress.Glyph:
    return compress.Glyph(char, mask.astype(bool))


class ProjectionBandTests(unittest.TestCase):
    def test_horizontal_bar_is_one_stroke(self):
        projection = np.zeros(30, dtype=int)
        projection[10:16] = 80
        bands = compress.projection_bands(projection, min_pixels=40, max_thickness=10)
        self.assertEqual(bands, [(10, 15, 80 * 6)])

    def test_thick_blob_is_rejected(self):
        projection = np.full(30, 80, dtype=int)
        bands = compress.projection_bands(projection, min_pixels=40, max_thickness=8)
        self.assertEqual(bands, [])

    def test_speck_below_min_span_is_ignored(self):
        projection = np.zeros(20, dtype=int)
        projection[4] = 5
        bands = compress.projection_bands(projection, min_pixels=40, max_thickness=8)
        self.assertEqual(bands, [])


class SyntheticPlacementTests(unittest.TestCase):
    def test_aligns_two_horizontal_bars(self):
        bar_a = np.zeros((40, 80), dtype=bool)
        bar_a[20:25, :] = True
        bar_b = np.zeros((40, 80), dtype=bool)
        bar_b[5:10, :] = True
        placements = compress.search(
            _glyph("A", bar_a),
            _glyph("B", bar_b),
            min_span=30,
            max_thick=8,
            max_cover=1.0,
        )
        self.assertTrue(placements)
        best = placements[0]
        # dx starts at a quarter of the glyph width, where the full-width bars
        # still overlap the most.
        self.assertEqual((best.dx, best.dy), (20, 15))
        self.assertEqual(best.stroke_count, 1)
        self.assertEqual(best.strokes[0].orientation, "H")
        self.assertEqual(best.overlap_pixels, 5 * 60)

    def test_solid_blocks_are_not_strokes(self):
        block = np.ones((30, 30), dtype=bool)
        stacked = compress.evaluate(
            _glyph("A", block),
            _glyph("B", block),
            8,
            0,
            min_span=10,
            max_thick=8,
            max_cover=1.0,
        )
        self.assertIsNone(stacked)

    def test_render_paints_both_glyphs(self):
        mask = np.zeros((4, 4), dtype=bool)
        mask[0:2, 0:2] = True
        image = compress.render_pair(_glyph("A", mask), _glyph("B", mask), dx=3, dy=1, margin=0)
        arr = np.asarray(image.convert("L"))
        self.assertGreater(int((arr > 0).sum()), 0)
        self.assertEqual(arr.shape[1], 4 + 3)
        self.assertEqual(arr.shape[0], 4 + 1)


class HeightWindowTests(unittest.TestCase):
    def test_window_is_centered_on_the_first_glyph(self):
        # Font size 220, ink height 199: (-220, 220) around the center row.
        y_low, y_high = compress.line_window(199, 220)
        self.assertEqual((y_low, y_high), (99 - 220, 99 + 220))
        self.assertTrue(compress.ink_inside_window(0, 199, y_low, y_high))
        # Side-by-side share the pair search kept (overlap 1016, dy=27).
        self.assertTrue(compress.ink_inside_window(27, 199, y_low, y_high))
        # Full-height stack above the first glyph, and the nest underneath.
        self.assertFalse(compress.ink_inside_window(-157, 199, y_low, y_high))
        self.assertFalse(compress.ink_inside_window(129, 199, y_low, y_high))

    def test_search_can_reject_a_vertical_shift(self):
        bar_a = np.zeros((40, 80), dtype=bool)
        bar_a[20:25, :] = True
        bar_b = np.zeros((40, 80), dtype=bool)
        bar_b[5:10, :] = True
        limited = compress.search(
            _glyph("A", bar_a),
            _glyph("B", bar_b),
            min_span=30,
            max_thick=8,
            max_cover=1.0,
            dy_max=5,
        )
        self.assertTrue(all(item.dy <= 5 for item in limited))
        self.assertFalse(any((item.dx, item.dy) == (0, 15) for item in limited))

    def test_line_prefers_an_in_window_stroke_over_a_staircase(self):
        width, height = 36, 24
        first = np.zeros((height, width), dtype=bool)
        second = np.zeros((height, width), dtype=bool)
        # Horizontal bars align only when the second glyph drops well below.
        first[18:21, :] = True
        second[2:5, :] = True
        # Vertical bars align on the same baseline once dx clears width/4.
        first[:, 20:23] = True
        second[:, 11:14] = True
        bound = 16
        placed = compress.place_line(
            [_glyph("高", first), _glyph("速", second)],
            bound=bound,
            min_span_ratio=0.2,
            max_thick=6,
            max_cover=0.8,
        )
        self.assertEqual(len(placed), 2)
        y_low, y_high = compress.line_window(height, bound)
        for item in placed:
            self.assertTrue(
                compress.ink_inside_window(item.y, item.glyph.height, y_low, y_high)
            )
        self.assertNotEqual(placed[1].y, 16)

    def test_line_throws_out_a_placement_that_only_fits_by_staircasing(self):
        width, height = 36, 24
        first = np.zeros((height, width), dtype=bool)
        second = np.zeros((height, width), dtype=bool)
        first[18:21, :] = True
        second[2:5, :] = True
        with self.assertRaises(RuntimeError):
            compress.place_line(
                [_glyph("高", first), _glyph("速", second)],
                bound=16,
                min_span_ratio=0.2,
                max_thick=6,
                max_cover=0.8,
            )


class EarlierOverlapTests(unittest.TestCase):
    def test_third_glyph_moves_off_the_one_two_back(self):
        width, height = 40, 16
        bar_rows = slice(6, 9)
        masks = []
        for _ in range(3):
            mask = np.zeros((height, width), dtype=bool)
            mask[bar_rows, :] = True
            masks.append(mask)
        placed = compress.place_line(
            [_glyph(char, mask) for char, mask in zip("高速素", masks)],
            bound=40,
            min_span_ratio=0.2,
            max_thick=6,
            max_cover=1.0,
        )
        self.assertEqual(len(placed), 3)
        # The second glyph still shares the bar with the first.
        self.assertEqual(placed[1].placement.earlier_overlap, 0)
        self.assertGreater(placed[1].x, 0)
        # Minimum dx would leave the third glyph covering the first. The
        # penalty moves it until those bars no longer touch.
        self.assertEqual(
            compress.placed_ink_overlap(placed[0], placed[2].glyph, placed[2].x, placed[2].y),
            0,
        )
        self.assertGreater(placed[2].x, placed[1].x)


class MinchoPairTests(unittest.TestCase):
    def test_first_two_template_characters_share_one_or_two_strokes(self):
        font = ImageFont.truetype(str(FONT), 160)
        text = compress.load_text(ROOT / "template.txt", 2)
        self.assertEqual(text, "高速")
        first = compress.rasterize(font, text[0])
        second = compress.rasterize(font, text[1])
        min_span = max(8, round(min(first.width, second.width) * 0.20))
        max_thick = max(4, round(160 * 0.08))
        placements = compress.search(
            first,
            second,
            min_span=min_span,
            max_thick=max_thick,
            max_cover=0.22,
        )
        self.assertTrue(placements)
        best = placements[0]
        self.assertIn(best.stroke_count, (1, 2))
        self.assertGreaterEqual(best.dx, first.width / 4)
        self.assertLess(best.dx, first.width)
        self.assertGreater(best.dy, -second.height)
        self.assertLess(best.dy, first.height)
        self.assertGreater(best.overlap_pixels, 100)
        self.assertLessEqual(best.cover, 0.22)
        self.assertLess(best.invasion, 0.45)
        self.assertGreater(best.stroke_pixels / best.overlap_pixels, 0.7)

        window = compress._overlap_window(first.mask, second.mask, best.dx, best.dy)
        self.assertIsNotNone(window)
        slice_a, slice_b = window
        self.assertEqual(int((slice_a & slice_b).sum()), best.overlap_pixels)


if __name__ == "__main__":
    unittest.main()
