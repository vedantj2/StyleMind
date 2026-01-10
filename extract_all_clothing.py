#!/usr/bin/env python
# -*- encoding: utf-8 -*-

"""
Extract all clothing pieces from human parsing results into separate images.
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from PIL import Image

# Ensure imports work when run from any cwd
sys.path.append(str(Path(__file__).resolve().parent))
from extract_clothing import extract_clothing_item, get_label_mapping


def extract_all_items(original_img_path, mask_path, output_dir, dataset='lip', 
                     background='transparent', exclude_background=True):
    """
    Extract all clothing items into separate images.
    
    Args:
        original_img_path: Path to original image
        mask_path: Path to parsing mask
        output_dir: Directory to save extracted images
        dataset: Dataset type ('lip', 'atr', or 'pascal')
        background: Background style ('transparent', 'white', 'black', 'original')
        exclude_background: Whether to skip background class (index 0)
    """
    # Get labels
    labels = get_label_mapping(dataset)
    
    # Load images
    original_img = Image.open(original_img_path).convert('RGB')
    mask_img = Image.open(mask_path)
    
    # Convert mask to numpy array
    if mask_img.mode == 'P':
        mask = np.array(mask_img)
    else:
        mask = np.array(mask_img.convert('L'))
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract each item
    extracted_count = 0
    for idx, label in enumerate(labels):
        if exclude_background and idx == 0:
            continue
        
        # Check if this class exists in the mask
        if np.any(mask == idx):
            # Extract this item
            result = extract_clothing_item(original_img, mask, [idx], background=background)
            
            # Save with sanitized filename
            safe_label = label.replace(' ', '_').replace('-', '_').lower()
            # Use PNG to avoid format issues; keeps transparency when requested
            output_path = os.path.join(output_dir, f"{idx:02d}_{safe_label}.png")
            result.save(output_path)
            print(f"Extracted: {label} -> {output_path}")
            extracted_count += 1
        else:
            print(f"Skipped: {label} (not found in mask)")
    
    print(f"\nExtracted {extracted_count} clothing items to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Extract all clothing pieces from human parsing results")
    parser.add_argument("--original-image", type=str, required=True, help="Path to original image")
    parser.add_argument("--parsing-mask", type=str, required=True, help="Path to parsing mask PNG")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for extracted images")
    parser.add_argument("--dataset", type=str, default='lip', choices=['lip', 'atr', 'pascal'],
                       help="Dataset type (default: lip)")
    parser.add_argument("--background", type=str, default='transparent', 
                       choices=['transparent', 'white', 'black', 'original'],
                       help="Background style (default: transparent)")
    parser.add_argument("--include-background", action='store_true',
                       help="Include background class in extraction")
    
    args = parser.parse_args()
    
    extract_all_items(
        args.original_image,
        args.parsing_mask,
        args.output_dir,
        dataset=args.dataset,
        background=args.background,
        exclude_background=not args.include_background
    )


if __name__ == '__main__':
    main()

