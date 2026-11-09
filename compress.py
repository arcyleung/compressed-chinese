#!/usr/bin/env python3
"""Compress CJK glyphs by sharing one or two broad strokes.

The next glyph is shifted over the previous one. A placement is kept when the
overlapping ink is one or two thin horizontal or vertical strokes, and neither
glyph loses so much ink that it stops being readable. The recognizability cap
is ``--max-cover``.

Offsets are in pixels of the tight ink bounding box. Positive ``dx`` moves the
incoming glyph right. Positive ``dy`` moves it down. Against the previous
glyph, ``dx`` is searched in ``[width / 4, width)`` and ``dy`` in
``(-height, height)``.

A line of more than two characters also has to stay inside one vertical band.
That band is ``(-H, H)`` centered on the first glyph, with ``H`` the font size.
Any placement whose ink leaves the band is thrown out, so later characters
cannot staircase away from the first one.

Ink shared with the previous glyph is the intended overlap. Ink shared with
any glyph further back (the one two behind, and older) is subtracted from the
score, so a new character cannot sit on top of an earlier one.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

ROOT = Path(__file__).resolve().parent

# Invasion is a 0–1 fraction of a glyph sitting in the shared rectangle without
# actually overlapping. Scale it into pixel units comparable to a short stroke
# so a crowded placement loses to a clean one of similar overlap.
INVASION_WEIGHT = 0.06
JUNK_WEIGHT = 1.0
# One pixel of ink shared with a glyph before the previous one cancels one
# point of pair score. The pair score is what the new glyph shares with the
# glyph it is attached to.
EARLIER_OVERLAP_WEIGHT = 1.0


@dataclass(frozen=True)
class Glyph:
    char: str
    mask: np.ndarray  # bool array, tight ink bounding box

    @property
    def height(self) -> int:
        return int(self.mask.shape[0])

    @property
    def width(self) -> int:
        return int(self.mask.shape[1])

    @property
    def ink(self) -> int:
        return int(self.mask.sum())


@dataclass(frozen=True)
class StrokeBand:
    orientation: str  # "H" or "V"
    thickness: int
    pixels: int


@dataclass(frozen=True)
class Placement:
    dx: int
    dy: int
    score: float
    overlap_pixels: int
    stroke_pixels: int
    junk_pixels: int
    stroke_count: int
    groups: int
    cover: float
    invasion: float
    strokes: tuple[StrokeBand, ...]
    earlier_overlap: int = 0


@dataclass(frozen=True)
class AnchoredGlyph:
    """A glyph placed in line coordinates. The first glyph sits at ``(0, 0)``."""

    glyph: Glyph
    x: int
    y: int
    placement: Placement | None = None


def rasterize(font: ImageFont.ImageFont, char: str, threshold: int = 128) -> Glyph:
    """Render one character to a binary mask cropped to its ink."""
    if len(char) != 1:
        raise ValueError(f"expected one character, got {char!r}")
    size = int(getattr(font, "size", 0) or 0)
    canvas = max(size * 3, 64)
    image = Image.new("L", (canvas, canvas), 0)
    ImageDraw.Draw(image).text((canvas // 6, canvas // 6), char, font=font, fill=255)
    arr = np.asarray(image)
    ink = arr > threshold
    ys, xs = np.nonzero(ink)
    if ys.size == 0:
        raise ValueError(f"no ink rendered for {char!r}")
    mask = ink[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    return Glyph(char, mask)


def projection_bands(
    projection: np.ndarray, min_pixels: int, max_thickness: int
) -> list[tuple[int, int, int]]:
    """Contiguous runs whose sum stays broad and whose thickness stays stroke-like.

    Returns ``(start, end, pixels)`` inclusive of ``end``. A run thicker than
    ``max_thickness`` is a blob, not a stroke, and is dropped.
    """
    found: list[tuple[int, int, int]] = []
    start: int | None = None
    pixels = 0
    for index, value in enumerate(np.asarray(projection)):
        if int(value) >= min_pixels:
            if start is None:
                start = index
                pixels = 0
            pixels += int(value)
        elif start is not None:
            thickness = index - start
            if thickness <= max_thickness:
                found.append((start, index - 1, pixels))
            start = None
            pixels = 0
    if start is not None:
        thickness = len(projection) - start
        if thickness <= max_thickness:
            found.append((start, len(projection) - 1, pixels))
    return found


def _overlap_window(
    a: np.ndarray, b: np.ndarray, dx: int, dy: int
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return the intersecting slices of ``a`` and ``b`` at this offset."""
    height_a, width_a = a.shape
    height_b, width_b = b.shape
    y0 = max(0, dy)
    y1 = min(height_a, dy + height_b)
    x0 = max(0, dx)
    x1 = min(width_a, dx + width_b)
    if y1 <= y0 or x1 <= x0:
        return None
    return (
        a[y0:y1, x0:x1],
        b[y0 - dy : y1 - dy, x0 - dx : x1 - dx],
    )


def evaluate(
    a: Glyph,
    b: Glyph,
    dx: int,
    dy: int,
    *,
    min_span: int,
    max_thick: int,
    max_cover: float,
    max_strokes: int = 2,
    max_groups: int = 4,
    dx_lims: tuple = None,
    dy_lims: tuple = None,
    allow_negative: bool = False,
) -> Placement | None:
    """Score one offset. ``None`` means the placement is not a candidate.

    The score follows the README: many overlapping pixels, concentrated in the
    fewest broad vertical or horizontal strokes, without swallowing either glyph.
    """

    if not (a.width / 4 <= dx < a.width and -b.height < dy < a.height):
        return None
    window = _overlap_window(a.mask, b.mask, dx, dy)
    if window is None:
        return None
    slice_a, slice_b = window
    overlap = slice_a & slice_b
    total = int(overlap.sum())
    if total == 0:
        return None
    cover = max(total / a.ink, total / b.ink)
    if cover > max_cover:
        return None

    horizontal = projection_bands(overlap.sum(axis=1), min_span, max_thick)
    vertical = projection_bands(overlap.sum(axis=0), min_span, max_thick)
    if not horizontal and not vertical:
        return None
    if len(horizontal) + len(vertical) > max_strokes:
        return None

    explained = np.zeros(overlap.shape, dtype=bool)
    strokes: list[StrokeBand] = []
    for start, end, pixels in horizontal:
        explained[start : end + 1] |= overlap[start : end + 1]
        strokes.append(StrokeBand("H", end - start + 1, pixels))
    for start, end, pixels in vertical:
        explained[:, start : end + 1] |= overlap[:, start : end + 1]
        strokes.append(StrokeBand("V", end - start + 1, pixels))

    stroke_pixels = int(explained.sum())
    junk = total - stroke_pixels
    labeled, group_count = ndimage.label(overlap)
    if group_count:
        sizes = ndimage.sum(overlap, labeled, index=range(1, group_count + 1))
        groups = int(sum(int(size) >= min_span for size in sizes))
    else:
        groups = 0
    if groups == 0 or groups > max_groups:
        return None

    invasion = max(
        (int(slice_a.sum()) - total) / a.ink,
        (int(slice_b.sum()) - total) / b.ink,
    )
    group_factor = 1.0 if len(strokes) == 1 else 0.8
    if groups > 2:
        group_factor *= 0.85
    score = (
        stroke_pixels * group_factor * (1.0 - invasion)
        - JUNK_WEIGHT * junk
        - INVASION_WEIGHT * (a.ink + b.ink) * invasion
    )
    if score <= 0 and not allow_negative:
        return None
    return Placement(
        dx=dx,
        dy=dy,
        score=float(score),
        overlap_pixels=total,
        stroke_pixels=stroke_pixels,
        junk_pixels=junk,
        stroke_count=len(strokes),
        groups=groups,
        cover=float(cover),
        invasion=float(invasion),
        strokes=tuple(strokes),
    )


def search(
    a: Glyph,
    b: Glyph,
    *,
    min_span: int,
    max_thick: int,
    max_cover: float,
    max_strokes: int = 2,
    dy_min: int | None = None,
    dy_max: int | None = None,
    allow_negative: bool = False,
) -> list[Placement]:
    """Brute-force in-bounds offsets and return viable placements, best first.

    ``dy_min`` and ``dy_max`` are inclusive extra limits. They drop shifts that
    would move a later glyph out of the line's height window. ``allow_negative``
    keeps structurally valid shares whose pair score is below zero, so a line
    can prefer one of them once overlap with earlier glyphs is penalized.
    """
    start = -(b.height - 1)
    stop = a.height
    if dy_min is not None:
        start = max(start, dy_min)
    if dy_max is not None:
        stop = min(stop, dy_max + 1)
    found: list[Placement] = []
    dx_start = math.ceil(a.width / 4)
    if start >= stop or dx_start >= a.width:
        return found
    for dy in range(start, stop):
        for dx in range(dx_start, a.width):
            placement = evaluate(
                a,
                b,
                dx,
                dy,
                min_span=min_span,
                max_thick=max_thick,
                max_cover=max_cover,
                max_strokes=max_strokes,
                allow_negative=allow_negative,
            )
            if placement is not None:
                found.append(placement)
    found.sort(key=lambda item: (item.score, item.stroke_pixels, -item.invasion), reverse=True)
    return found


def select_diverse(placements: list[Placement], count: int, radius: int) -> list[Placement]:
    """Keep the highest-scoring placements whose offsets are not near each other."""
    chosen: list[Placement] = []
    for placement in placements:
        if len(chosen) >= count:
            break
        if any(
            abs(placement.dx - other.dx) < radius and abs(placement.dy - other.dy) < radius
            for other in chosen
        ):
            continue
        chosen.append(placement)
    return chosen


def line_window(first_height: int, bound: int) -> tuple[int, int]:
    """Exclusive ink-row limits ``(-H, H)`` centered on the first glyph.

    Row 0 is the top of the first glyph. A row ``r`` is inside when
    ``-bound < r - center < bound``.
    """
    if bound <= 0:
        raise ValueError(f"height bound must be positive, got {bound}")
    if first_height <= 0:
        raise ValueError(f"first glyph has no height, got {first_height}")
    center = first_height // 2
    return center - bound, center + bound


def ink_inside_window(y: int, height: int, y_low: int, y_high: int) -> bool:
    """True when every ink row of a glyph at ``y`` lies in ``(y_low, y_high)``."""
    if height <= 0:
        return False
    return y > y_low and y + height - 1 < y_high


def placed_ink_overlap(prior: AnchoredGlyph, glyph: Glyph, x: int, y: int) -> int:
    """Ink pixels ``glyph`` at ``(x, y)`` shares with an already placed glyph."""
    prior_mask = prior.glyph.mask
    incoming = glyph.mask
    y0 = max(prior.y, y)
    y1 = min(prior.y + prior_mask.shape[0], y + incoming.shape[0])
    x0 = max(prior.x, x)
    x1 = min(prior.x + prior_mask.shape[1], x + incoming.shape[1])
    if y1 <= y0 or x1 <= x0:
        return 0
    shared = prior_mask[y0 - prior.y : y1 - prior.y, x0 - prior.x : x1 - prior.x]
    shared = shared & incoming[y0 - y : y1 - y, x0 - x : x1 - x]
    return int(shared.sum())


def _penalize_earlier(
    placement: Placement, extra_pixels: int
) -> Placement:
    """Subtract ink shared with glyphs before the one this placement attaches to."""
    score = placement.score - EARLIER_OVERLAP_WEIGHT * extra_pixels
    return replace(placement, score=float(score), earlier_overlap=extra_pixels)


def _dy_inside_window(
    anchor_y: int, height: int, y_low: int, y_high: int
) -> tuple[int, int]:
    """Inclusive ``dy`` range that keeps the incoming glyph inside the window."""
    dy_min = y_low - anchor_y + 1
    dy_max = y_high - height - anchor_y
    return dy_min, dy_max


def place_line(
    glyphs: list[Glyph],
    *,
    bound: int,
    min_span_ratio: float,
    max_thick: int,
    max_cover: float,
    max_strokes: int = 2,
) -> list[AnchoredGlyph]:
    """Place glyphs left to right. Drop any shift that leaves ``(-bound, bound)``.

    Each glyph is scored against the one before it. The first glyph is anchored
    at ``(0, 0)``, and the window is centered on that glyph.
    """
    if not glyphs:
        raise ValueError("need at least one glyph")
    y_low, y_high = line_window(glyphs[0].height, bound)
    if not ink_inside_window(0, glyphs[0].height, y_low, y_high):
        raise RuntimeError(
            f"{glyphs[0].char} is {glyphs[0].height}px tall and does not fit "
            f"in the height window ({y_low}, {y_high})"
        )
    placed = [AnchoredGlyph(glyphs[0], 0, 0, None)]
    for glyph in glyphs[1:]:
        anchor = placed[-1]
        earlier = placed[:-1]
        min_span = max(8, round(min(anchor.glyph.width, glyph.width) * min_span_ratio))
        dy_min, dy_max = _dy_inside_window(anchor.y, glyph.height, y_low, y_high)
        found = search(
            anchor.glyph,
            glyph,
            min_span=min_span,
            max_thick=max_thick,
            max_cover=max_cover,
            max_strokes=max_strokes,
            dy_min=dy_min,
            dy_max=dy_max,
            allow_negative=bool(earlier),
        )
        found = [
            item
            for item in found
            if ink_inside_window(anchor.y + item.dy, glyph.height, y_low, y_high)
        ]
        if earlier:
            adjusted = []
            for item in found:
                x = anchor.x + item.dx
                y = anchor.y + item.dy
                extra = sum(placed_ink_overlap(prior, glyph, x, y) for prior in earlier)
                adjusted.append(_penalize_earlier(item, extra))
            adjusted.sort(
                key=lambda item: (
                    item.score,
                    -item.earlier_overlap,
                    item.stroke_pixels,
                    -item.invasion,
                ),
                reverse=True,
            )
            found = adjusted
        if not found:
            raise RuntimeError(
                f"no readable overlap for {anchor.glyph.char}{glyph.char} "
                f"inside the height window ({y_low}, {y_high})"
            )
        best = found[0]
        placed.append(
            AnchoredGlyph(glyph, anchor.x + best.dx, anchor.y + best.dy, best)
        )
    return placed


def best_second_below(placements: list[Placement], first: Glyph) -> Placement | None:
    """Best placement that drops the second glyph under the first.

    A full-width shared rail often ranks higher, but the reference poster nests
    the second character underneath. This is that nest, kept for comparison.
    """
    pool = [
        item
        for item in placements
        if item.dy > first.height * 0.45 and item.dx < first.width * 0.6
    ]
    return pool[0] if pool else None


def composite_box(a: Glyph, b: Glyph, dx: int, dy: int) -> tuple[int, int, int, int]:
    """Return ``(x0, y0, width, height)`` of the union, with ``a`` at ``(-x0, -y0)``."""
    x0 = min(0, dx)
    y0 = min(0, dy)
    width = max(a.width, dx + b.width) - x0
    height = max(a.height, dy + b.height) - y0
    return x0, y0, width, height


def render_pair(a: Glyph, b: Glyph, dx: int, dy: int, margin: int = 28) -> Image.Image:
    """White glyphs on black. Overlapping ink stays white, so a shared stroke reads as one."""
    x0, y0, width, height = composite_box(a, b, dx, dy)
    canvas = np.zeros((height + margin * 2, width + margin * 2), dtype=np.uint8)
    ay, ax = margin - y0, margin - x0
    canvas[ay : ay + a.height, ax : ax + a.width][a.mask] = 255
    by, bx = margin + dy - y0, margin + dx - x0
    canvas[by : by + b.height, bx : bx + b.width][b.mask] = 255
    return Image.fromarray(canvas, mode="L").convert("RGB")


def render_overlap(a: Glyph, b: Glyph, dx: int, dy: int, margin: int = 28) -> Image.Image:
    """Diagnostic: first glyph red, second cyan, shared strokes white."""
    x0, y0, width, height = composite_box(a, b, dx, dy)
    canvas = np.zeros((height + margin * 2, width + margin * 2, 3), dtype=np.uint8)
    ay, ax = margin - y0, margin - x0
    by, bx = margin + dy - y0, margin + dx - x0
    canvas[ay : ay + a.height, ax : ax + a.width][a.mask] = (230, 70, 70)
    paint = canvas[by : by + b.height, bx : bx + b.width]
    paint[b.mask] = (70, 190, 230)
    both = np.zeros(canvas.shape[:2], dtype=bool)
    both[ay : ay + a.height, ax : ax + a.width] |= a.mask
    second = np.zeros_like(both)
    second[by : by + b.height, bx : bx + b.width] |= b.mask
    canvas[both & second] = (255, 255, 255)
    return Image.fromarray(canvas, mode="RGB")


_ROLE_COLORS = (
    (235, 235, 235),
    (255, 170, 60),
    (90, 190, 255),
    (160, 230, 120),
    (255, 120, 160),
    (190, 150, 255),
    (255, 220, 90),
    (120, 220, 210),
    (255, 140, 90),
    (170, 170, 255),
    (200, 255, 180),
)


def render_line(placed: list[AnchoredGlyph], margin: int = 28, roles: bool = False) -> Image.Image:
    """Draw a placed line. Shared ink stays white.

    With ``roles``, each glyph has its own color until it overlaps another,
    which is painted white.
    """
    if not placed:
        raise ValueError("no glyphs to draw")
    x0 = min(item.x for item in placed)
    y0 = min(item.y for item in placed)
    x1 = max(item.x + item.glyph.width for item in placed)
    y1 = max(item.y + item.glyph.height for item in placed)
    height = y1 - y0 + margin * 2
    width = x1 - x0 + margin * 2
    if not roles:
        canvas = np.zeros((height, width), dtype=np.uint8)
        for item in placed:
            top = margin + item.y - y0
            left = margin + item.x - x0
            view = canvas[top : top + item.glyph.height, left : left + item.glyph.width]
            view[item.glyph.mask] = 255
        return Image.fromarray(canvas, mode="L").convert("RGB")
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    occupied = np.zeros((height, width), dtype=bool)
    for index, item in enumerate(placed):
        top = margin + item.y - y0
        left = margin + item.x - x0
        region = canvas[top : top + item.glyph.height, left : left + item.glyph.width]
        occ = occupied[top : top + item.glyph.height, left : left + item.glyph.width]
        both = item.glyph.mask & occ
        fresh = item.glyph.mask & ~occ
        region[fresh] = _ROLE_COLORS[index % len(_ROLE_COLORS)]
        region[both] = (255, 255, 255)
        occ[item.glyph.mask] = True
    return Image.fromarray(canvas, mode="RGB")


def _caption_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def contact_sheet(a: Glyph, b: Glyph, placements: list[Placement]) -> Image.Image:
    """Grid of candidate renderings with their offsets."""
    if not placements:
        raise ValueError("no placements to draw")
    thumb_w, thumb_h = 320, 420
    label_h = 36
    pad = 16
    columns = 3 if len(placements) > 1 else 1
    rows = (len(placements) + columns - 1) // columns
    sheet = Image.new(
        "RGB",
        (columns * thumb_w + (columns + 1) * pad, rows * (thumb_h + label_h) + (rows + 1) * pad),
        (0, 0, 0),
    )
    draw = ImageDraw.Draw(sheet)
    font = _caption_font(16)
    for index, placement in enumerate(placements):
        row, column = divmod(index, columns)
        cell = render_pair(a, b, placement.dx, placement.dy, margin=16)
        cell.thumbnail((thumb_w - 8, thumb_h - 8))
        x = pad + column * (thumb_w + pad) + (thumb_w - cell.width) // 2
        y = pad + row * (thumb_h + label_h + pad)
        sheet.paste(cell, (x, y))
        orients = "".join(stroke.orientation for stroke in placement.strokes)
        caption = (
            f"dx={placement.dx} dy={placement.dy}  {orients}  "
            f"overlap={placement.overlap_pixels}"
        )
        draw.text((pad + column * (thumb_w + pad) + 8, y + thumb_h - 4), caption, fill=(220, 220, 220), font=font)
    return sheet


def placement_record(placement: Placement) -> dict:
    record = asdict(placement)
    record["score"] = round(placement.score, 1)
    record["cover"] = round(placement.cover, 4)
    record["invasion"] = round(placement.invasion, 4)
    return record


def load_text(path: Path, chars: int) -> str:
    raw = path.read_text(encoding="utf-8")
    text = "".join(raw.split())
    if len(text) < 2:
        raise ValueError(f"{path} needs at least two characters")
    return text[:chars]


def _line_record(item: AnchoredGlyph) -> dict:
    record = {"char": item.glyph.char, "x": item.x, "y": item.y}
    if item.placement is not None:
        record.update(placement_record(item.placement))
    return record


def run_line(args: argparse.Namespace, text: str, font: ImageFont.ImageFont) -> None:
    """Place ``text`` inside the global height window and write the line."""
    glyphs = [rasterize(font, char) for char in text]
    bound = args.size if args.height_bound is None else args.height_bound
    max_thick = max(4, round(args.size * args.max_thickness))
    y_low, y_high = line_window(glyphs[0].height, bound)
    print(
        f"placing {text} ({len(text)} characters), "
        f"height window ({y_low}, {y_high}) = (-{bound}, {bound}) "
        f"around the center of {text[0]}"
    )
    for glyph in glyphs:
        print(f"  {glyph.char} {glyph.width}x{glyph.height} ink={glyph.ink}")
    placed = place_line(
        glyphs,
        bound=bound,
        min_span_ratio=args.min_span,
        max_thick=max_thick,
        max_cover=args.max_cover,
        max_strokes=args.max_strokes,
    )
    for item in placed:
        if item.placement is None:
            print(f"{item.glyph.char} x={item.x} y={item.y}")
            continue
        orients = ",".join(
            f"{stroke.orientation}{stroke.thickness}px/{stroke.pixels}"
            for stroke in item.placement.strokes
        )
        print(
            f"{item.glyph.char} x={item.x} y={item.y} "
            f"dx={item.placement.dx} dy={item.placement.dy} "
            f"score={item.placement.score:.1f} {orients} "
            f"groups={item.placement.groups} cover={item.placement.cover:.3f} "
            f"earlier={item.placement.earlier_overlap}"
        )
        ink_top = item.y
        ink_bottom = item.y + item.glyph.height - 1
        if not ink_inside_window(item.y, item.glyph.height, y_low, y_high):
            raise RuntimeError(
                f"{item.glyph.char} ink rows {ink_top}..{ink_bottom} "
                f"left the window ({y_low}, {y_high})"
            )
    ink_top = min(item.y for item in placed)
    ink_bottom = max(item.y + item.glyph.height - 1 for item in placed)
    print(f"line ink rows {ink_top}..{ink_bottom} inside ({y_low}, {y_high})")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    render_line(placed).save(out / "line.png")
    render_line(placed, roles=True).save(out / "line-roles.png")
    payload = {
        "text": text,
        "font": str(args.font),
        "size": args.size,
        "height_bound": bound,
        "window": [y_low, y_high],
        "max_cover": args.max_cover,
        "max_thickness_px": max_thick,
        "placements": [_line_record(item) for item in placed],
    }
    (out / "placements.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {out / 'line.png'}")
    print(f"wrote {out / 'line-roles.png'}")
    print(f"wrote {out / 'placements.json'}")


def run(args: argparse.Namespace) -> None:
    if args.text is not None:
        text = "".join(args.text.split())
        text = text[: args.chars] if args.chars else text
    else:
        text = load_text(Path(args.template), args.chars)
    if len(text) < 2:
        raise SystemExit("need at least two characters")

    font = ImageFont.truetype(args.font, args.size)
    if len(text) > 2:
        run_line(args, text, font)
        return

    pair = text
    first = rasterize(font, pair[0])
    second = rasterize(font, pair[1])
    min_span = max(8, round(min(first.width, second.width) * args.min_span))
    max_thick = max(4, round(args.size * args.max_thickness))
    print(
        f"{pair[0]} {first.width}x{first.height} ink={first.ink} | "
        f"{pair[1]} {second.width}x{second.height} ink={second.ink}"
    )
    print(
        f"search dx in [0, {first.width}), dy in (-{second.height}, {first.height}), "
        f"min_span={min_span}px, max_thickness={max_thick}px, max_cover={args.max_cover}"
    )
    placements = search(
        first,
        second,
        min_span=min_span,
        max_thick=max_thick,
        max_cover=args.max_cover,
        max_strokes=args.max_strokes,
    )
    if not placements:
        raise SystemExit(
            "no placement kept both glyphs readable. Raise --max-cover or lower --min-span."
        )
    diverse = select_diverse(placements, args.top, radius=max(12, args.size // 3))
    below = best_second_below(placements, first)
    if below is not None and all(below.dx != item.dx or below.dy != item.dy for item in diverse):
        diverse.append(below)
    best = diverse[0]
    print(
        f"{len(placements)} viable placements. "
        f"best dx={best.dx} dy={best.dy} score={best.score:.1f} "
        f"strokes={best.stroke_count} groups={best.groups} "
        f"overlap={best.overlap_pixels} cover={best.cover:.3f}"
    )
    for index, placement in enumerate(diverse, start=1):
        orients = ",".join(
            f"{stroke.orientation}{stroke.thickness}px/{stroke.pixels}" for stroke in placement.strokes
        )
        print(
            f"  {index}. dx={placement.dx:4d} dy={placement.dy:4d} "
            f"score={placement.score:7.1f} {orients} groups={placement.groups} "
            f"cover={placement.cover:.3f}"
        )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    render_pair(first, second, best.dx, best.dy).save(out / "best.png")
    render_overlap(first, second, best.dx, best.dy).save(out / "best-overlap.png")
    if below is not None:
        render_pair(first, second, below.dx, below.dy).save(out / "second-below.png")
        render_overlap(first, second, below.dx, below.dy).save(out / "second-below-overlap.png")
    contact_sheet(first, second, diverse).save(out / "candidates.png")
    payload = {
        "text": pair,
        "font": str(args.font),
        "size": args.size,
        "max_cover": args.max_cover,
        "min_span_px": min_span,
        "max_thickness_px": max_thick,
        "best": placement_record(best),
        "second_below": placement_record(below) if below is not None else None,
        "candidates": [placement_record(item) for item in diverse],
    }
    (out / "placements.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out / 'best.png'}")
    print(f"wrote {out / 'best-overlap.png'}")
    if below is not None:
        print(
            f"second glyph tucked under the first at dx={below.dx} dy={below.dy} "
            f"(score={below.score:.1f}, lower than the shared rail)"
        )
        print(f"wrote {out / 'second-below.png'}")
        print(f"wrote {out / 'second-below-overlap.png'}")
    print(f"wrote {out / 'candidates.png'}")
    print(f"wrote {out / 'placements.json'}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Overlap two CJK glyphs along shared strokes.")
    parser.add_argument("--text", default=None, help="characters to place (default: template file)")
    parser.add_argument("--template", default=str(ROOT / "template.txt"), help="text source")
    parser.add_argument(
        "--chars",
        type=int,
        default=2,
        help="how many characters to place (more than two uses the height window)",
    )
    parser.add_argument(
        "--height-bound",
        type=int,
        default=None,
        help="half-height H of the global ink window (-H, H) centered on the first glyph (default: font size)",
    )
    parser.add_argument("--font", default=str(ROOT / "ZenOldMincho-Regular.ttf"))
    parser.add_argument("--size", type=int, default=220, help="font size in pixels")
    parser.add_argument(
        "--max-cover",
        type=float,
        default=0.22,
        help="max fraction of either glyph's ink that may overlap (recognizability)",
    )
    parser.add_argument(
        "--min-span",
        type=float,
        default=0.20,
        help="minimum stroke-overlap length as a fraction of the narrower glyph",
    )
    parser.add_argument(
        "--max-thickness",
        type=float,
        default=0.08,
        help="maximum stroke-band thickness as a fraction of the font size",
    )
    parser.add_argument("--max-strokes", type=int, default=2, help="maximum shared strokes (1 or 2)")
    parser.add_argument("--top", type=int, default=6, help="diverse candidates to render")
    parser.add_argument("--out", default=str(ROOT / "output"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
