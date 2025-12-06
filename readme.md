# Multi Particle Texture Combiner

## Description
Combine 2 textures : one as mask the other as texture.

The result is a pre-baked texture atlas that achieves the WoD+ multitexture
look without needing Cata+ shader support.

## Usage:

### Script execution
    python3 Multi_Particle_Texture_Combiner.py <mask.blp/png> <color.blp/png> <output.png> [options]
    
** Options:**
    --grid N         Output atlas grid NxN (default: 8, giving 64 frames)
    --cell-size N    Size of each cell in pixels (default: 256)
    --mask-grid N    If mask is 1x1 or 2x2 etc pattern (default 0, try to autodetect)
    --mode MODE      Compositing mode: multiply, screen, overlay (default: multiply)
    --variation      Add random variation to color sampling per frame
    --preview        Also generate a preview strip
    --seed N         Use a seed number to replicate same output texture, else it is random

Unrecommended to generate texture higher than 1024 pixel per side, so use --cell-size to avoid that
** Example: **
    
    py Multi_Particle_Texture_Combiner.py fire_2x2_sharp_mod4x.png fire_bright_mod2x_a.png test2.png --mask-grid 2 --grid 8 --cell-size 128 --seed 60

### M2 Edit
    When done, you can now:
    1. Convert the baked texture to BLP format
    2. Update your .M2 particle to use this texture
       - Check that texture index point to that specific baked texture
       - Set GeometryModel_Length to 1
       - .m2 EOL add a 4 bytes empty chunk
       - Set GeometryModel_Offset to the start of that previous chunk added
       - Set RecursionModel_Length to 1
       - .m2 EOL add a new 4 bytes empty chunk
       - Set RecursionModel_Offset to the start of the second chunk added
       - Set Texture_dimensions_rows and Texture_dimensions_columns to new value: {args.grid}x{args.grid}
       - Check particle_Flag and blending used")
    3. repeat for other particules using multi-textures, you can reuse the same chunk added if no extra setting")

##  Requirements:
    Python 3
    pip install Pillow