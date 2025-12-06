#!/usr/bin/env python3
"""
Multi Particle Texture Combiner

Combine 2 textures : one as mask the other as texture.

The result is a pre-baked texture atlas that achieves the WoD+ multitexture
look without needing Cata+ shader support.

Usage:
    python3 Multi_Particle_Texture_Combiner.py <mask.blp/png> <color.blp/png> <output.png> [options]
    
Options:
    --grid N         Output atlas grid NxN (default: 8, giving 64 frames)
    --cell-size N    Size of each cell in pixels (default: 256)
    --mask-grid N    If mask is 1x1 or 2x2 etc pattern (default 0, try to autodetect)
    --mode MODE      Compositing mode: multiply, screen, overlay (default: multiply)
    --variation      Add random variation to color sampling per frame
    --preview        Also generate a preview strip
    --seed N         Use a seed number to replicate same output texture, else it is random

Unrecommended to generate texture higher than 1024 pixel per side, so use --cell-size to avoid that
Example:
    
    py Multi_Particle_Texture_Combiner.py fire_2x2_sharp_mod4x.png fire_bright_mod2x_a.png test2.png --mask-grid 2 --grid 8 --cell-size 128 --seed 60

Requirements:
    pip install Pillow
"""

import argparse
import os
import sys
import random
from pathlib import Path

try:
    from PIL import Image, ImageEnhance, ImageFilter
    HAS_PIL = True
except ImportError:
    print("ERROR: Pillow is required. Install with:")
    print("  pip install Pillow --break-system-packages")
    sys.exit(1)


def detect_grid_size(image):
    """Detect if image is an atlas and return grid dimensions"""
    w, h = image.size
    
    if w != h:
        return 1, 1  # Not square, assume single image
    
    # Common atlas sizes
    for grid in [8, 4, 2]:
        cell_size = w // grid
        if cell_size >= 32 and w % grid == 0:
            return grid, grid
    
    return 1, 1


def extract_cells(image, grid_rows, grid_cols):
    """Extract individual cells from an atlas image"""
    w, h = image.size
    cell_w = w // grid_cols
    cell_h = h // grid_rows
    
    cells = []
    for row in range(grid_rows):
        for col in range(grid_cols):
            cell = image.crop((
                col * cell_w,
                row * cell_h,
                (col + 1) * cell_w,
                (row + 1) * cell_h
            ))
            cells.append(cell)
    
    return cells, cell_w, cell_h


def composite_multiply(mask, color):
    """Composite using multiply blend mode"""
    # Ensure same size
    if mask.size != color.size:
        color = color.resize(mask.size, Image.LANCZOS)
    
    result = Image.new('RGBA', mask.size)
    
    mask_data = mask.load()
    color_data = color.load()
    result_data = result.load()
    
    for y in range(mask.size[1]):
        for x in range(mask.size[0]):
            m = mask_data[x, y]
            c = color_data[x, y]
            
            # Get mask luminance/alpha
            if len(m) == 4:
                m_intensity = (m[0] + m[1] + m[2]) / (3 * 255) * (m[3] / 255)
            else:
                m_intensity = (m[0] + m[1] + m[2]) / (3 * 255)
            
            # Multiply color by mask intensity
            r = int(c[0] * m_intensity)
            g = int(c[1] * m_intensity)
            b = int(c[2] * m_intensity)
            a = int(255 * m_intensity)
            
            result_data[x, y] = (r, g, b, a)
    
    return result


def composite_screen(mask, color):
    """Composite using screen blend mode (brighter result)"""
    if mask.size != color.size:
        color = color.resize(mask.size, Image.LANCZOS)
    
    result = Image.new('RGBA', mask.size)
    
    mask_data = mask.load()
    color_data = color.load()
    result_data = result.load()
    
    for y in range(mask.size[1]):
        for x in range(mask.size[0]):
            m = mask_data[x, y]
            c = color_data[x, y]
            
            # Screen blend: 1 - (1-a)(1-b)
            r = int(255 - ((255 - m[0]) * (255 - c[0]) / 255))
            g = int(255 - ((255 - m[1]) * (255 - c[1]) / 255))
            b = int(255 - ((255 - m[2]) * (255 - c[2]) / 255))
            
            # Alpha from mask
            a = m[3] if len(m) == 4 else int((m[0] + m[1] + m[2]) / 3)
            
            result_data[x, y] = (r, g, b, a)
    
    return result


def composite_overlay(mask, color):
    """Composite using overlay blend mode"""
    if mask.size != color.size:
        color = color.resize(mask.size, Image.LANCZOS)
    
    result = Image.new('RGBA', mask.size)
    
    mask_data = mask.load()
    color_data = color.load()
    result_data = result.load()
    
    for y in range(mask.size[1]):
        for x in range(mask.size[0]):
            m = mask_data[x, y]
            c = color_data[x, y]
            
            out = []
            for i in range(3):
                if m[i] < 128:
                    out.append(int(2 * m[i] * c[i] / 255))
                else:
                    out.append(int(255 - 2 * (255 - m[i]) * (255 - c[i]) / 255))
            
            a = m[3] if len(m) == 4 else 255
            result_data[x, y] = (out[0], out[1], out[2], a)
    
    return result


def sample_color_with_offset(color_img, offset_x, offset_y, size):
    """Sample a region from color image with offset, tiling as needed"""
    w, h = color_img.size
    result = Image.new('RGBA', size)
    
    for y in range(size[1]):
        for x in range(size[0]):
            src_x = (x + offset_x) % w
            src_y = (y + offset_y) % h
            result.putpixel((x, y), color_img.getpixel((src_x, src_y)))
    
    return result


def create_composite_atlas(mask_path, color_path, output_path, 
                           grid_size=8, cell_size=256, mode='multiply', 
                           add_variation=True, mask_grid_override=None):
    """
    Create a composite texture atlas from mask and color textures
    """
    print(f"Loading mask: {mask_path}")
    mask_img = Image.open(mask_path).convert('RGBA')
    
    print(f"Loading color: {color_path}")
    color_img = Image.open(color_path).convert('RGBA')
    
    print(f"Mask size: {mask_img.size}")
    print(f"Color size: {color_img.size}")
    
    # Detect or use override for mask atlas grid
    if mask_grid_override:
        mask_grid_rows, mask_grid_cols = mask_grid_override, mask_grid_override
        print(f"Using mask grid override: {mask_grid_rows}x{mask_grid_cols}")
    else:
        mask_grid_rows, mask_grid_cols = detect_grid_size(mask_img)
        print(f"Detected mask grid: {mask_grid_rows}x{mask_grid_cols}")
    
    # Extract mask cells
    mask_cells, mask_cell_w, mask_cell_h = extract_cells(
        mask_img, mask_grid_rows, mask_grid_cols
    )
    print(f"Extracted {len(mask_cells)} mask cells ({mask_cell_w}x{mask_cell_h})")
    
    # Create output atlas
    output_size = cell_size * grid_size
    output = Image.new('RGBA', (output_size, output_size), (0, 0, 0, 0))
    
    total_frames = grid_size * grid_size
    
    print(f"Creating {total_frames} composited frames ({cell_size}x{cell_size} each)...")
    
    # Select composite function
    composite_func = {
        'multiply': composite_multiply,
        'screen': composite_screen,
        'overlay': composite_overlay,
    }.get(mode, composite_multiply)
    
    for frame_idx in range(total_frames):
        # Get mask cell (cycle through available)
        mask_cell = mask_cells[frame_idx % len(mask_cells)]
        
        # Resize mask cell to target size
        if mask_cell.size != (cell_size, cell_size):
            mask_cell = mask_cell.resize((cell_size, cell_size), Image.LANCZOS)
        
        # Get color sample
        if add_variation:
            # Random offset for variety
            offset_x = random.randint(0, color_img.size[0] - 1)
            offset_y = random.randint(0, color_img.size[1] - 1)
            color_sample = sample_color_with_offset(
                color_img, offset_x, offset_y, (cell_size, cell_size)
            )
        else:
            # Simple resize/tile
            color_sample = color_img.resize((cell_size, cell_size), Image.LANCZOS)
        
        # Composite
        composited = composite_func(mask_cell, color_sample)
        
        # Place in output atlas
        out_row = frame_idx // grid_size
        out_col = frame_idx % grid_size
        output.paste(composited, (out_col * cell_size, out_row * cell_size))
    
    # Save output
    output.save(output_path)
    print(f"Saved: {output_path}")
    print(f"Output size: {output_size}x{output_size}")
    
    return output


def create_preview(atlas, output_path, frames_to_show=8):
    """Create a horizontal strip preview"""
    grid_size = int(atlas.size[0] ** 0.5 // (atlas.size[0] / 8))  # Estimate grid
    cell_size = atlas.size[0] // 8  # Assume 8x8 grid
    
    preview_h = cell_size
    preview_w = cell_size * frames_to_show
    preview = Image.new('RGBA', (preview_w, preview_h), (32, 32, 32, 255))
    
    for i in range(frames_to_show):
        row = i // 8
        col = i % 8
        cell = atlas.crop((
            col * cell_size,
            row * cell_size,
            (col + 1) * cell_size,
            (row + 1) * cell_size
        ))
        preview.paste(cell, (i * cell_size, 0), cell)
    
    preview.save(output_path)
    print(f"Preview saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Create WotLK-compatible baked multi-textures for particle',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    # Basic usage
    python3 Multi_Particle_Texture_Combiner.py <mask.blp/png> <color.blp/png> <output.png> [options]
"""
    )
    
    parser.add_argument('mask', help='Path to mask/smoke texture')
    parser.add_argument('color', help='Path to color/fire texture')  
    parser.add_argument('output', help='Output path for composite atlas')
    parser.add_argument('--grid', type=int, default=8, help='Output grid size NxN (default: 8)')
    parser.add_argument('--cell-size', type=int, default=256, help='Cell size in pixels (default: 256)')
    parser.add_argument('--mask-grid', type=int, default=0, help='Mask grid size (0=auto-detect)')
    parser.add_argument('--mode', choices=['multiply', 'screen', 'overlay'], default='multiply',
                        help='Blend mode (default: multiply)')
    parser.add_argument('--no-variation', action='store_true', help='Disable random color sampling')
    parser.add_argument('--preview', action='store_true', help='Generate preview strip')
    parser.add_argument('--seed', type=int, help='Random seed for reproducible results')
    
    args = parser.parse_args()
    
    if args.seed is not None:
        random.seed(args.seed)
    
    print("=" * 60)
    print("Multi Particle Texture Combiner")
    print("=" * 60)
    
    atlas = create_composite_atlas(
        args.mask,
        args.color,
        args.output,
        grid_size=args.grid,
        cell_size=args.cell_size,
        mode=args.mode,
        add_variation=not args.no_variation,
        mask_grid_override=args.mask_grid if args.mask_grid > 0 else None
    )
    
    if args.preview:
        preview_path = Path(args.output).stem + '_preview.png'
        create_preview(atlas, preview_path)
    
    print("\nDone! You can now:")
    print(f"  1. Convert {args.output} to BLP format")
    print(f"  2. Update your .M2 particle to use this texture")
    print(f"     - Check that texture index point to that specific baked texture")
    print(f"     - Set GeometryModel_Length to 1")
    print(f"     - .m2 EOL add a 4 bytes empty chunk")
    print(f"     - Set GeometryModel_Offset to the start of that previous chunk added")
    print(f"     - Set RecursionModel_Length to 1")
    print(f"     - .m2 EOL add a new 4 bytes empty chunk")
    print(f"     - Set RecursionModel_Offset to the start of the second chunk added")
    print(f"     - Set Texture_dimensions_rows and Texture_dimensions_columns to new value: {args.grid}x{args.grid}")
    print(f"     - Check particle_Flag and blending used")
    print(f"  3. repeat for other particules using multi-textures, you can reuse the same chunk added if no extra setting")


if __name__ == '__main__':
    main()