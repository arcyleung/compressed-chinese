# Compressed Chinese/ Compressed Kanji

This tool renders Chinese/Kanji characters as glyphs, identifies strokes which are adjacent/ compressible from overlapping (adjustable by a factor, such that the original glyph is recognizable), and outputs the (x,y) offset of the next character to produce a "visually compressed" sentence

## How it works

- A raster of the individual glyph is taken, starting from the first and second characters (高速)
- Assuming left-to-right (LTR) reading order, the second character is constrained to be placed (0, W) to the right of the first character, and (-H, H) where W denotes width and H denotes height of the character in pixels.
- The placement may be randomly generated at first, or bruteforced generate all placements.
- The placement with the maximum number of overlapping pixels with the least number of disconnected groups of pixels are viable candidates (to select candidates which have a broad verical/ horizontal stroke overlapping, which tends to be more readable)
- Ideally, only 1-2 strokes (vertical or horizontal) should be overlapping to produce this "compressed" effect.
- A number of candidates may be generated for evaluation.