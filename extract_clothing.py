#!/usr/bin/env python
# -*- encoding: utf-8 -*-

"""
Extract specific clothing pieces from human parsing results.
"""

import os
import argparse
import numpy as np
from PIL import Image

# LIP dataset labels (pixel value = index in this list)
LIP_LABELS = ['Background', 'Hat', 'Hair', 'Glove', 'Sunglasses', 'Upper-clothes', 'Dress', 'Coat',
              'Socks', 'Pants', 'Jumpsuits', 'Scarf', 'Skirt', 'Face', 'Left-arm', 'Right-arm',
              'Left-leg', 'Right-leg', 'Left-shoe', 'Right-shoe']

# ATR dataset labels
ATR_LABELS = ['Background', 'Hat', 'Hair', 'Sunglasses', 'Upper-clothes', 'Skirt', 'Pants', 'Dress', 'Belt',
              'Left-shoe', 'Right-shoe', 'Face', 'Left-leg', 'Right-leg', 'Left-arm', 'Right-arm', 'Bag', 'Scarf']

# Pascal-Person-Part labels
PASCAL_LABELS = ['Background', 'Head', 'Torso', 'Upper Arms', 'Lower Arms', 'Upper Legs', 'Lower Legs']


def get_label_mapping(dataset):
    """Get label mapping for the specified dataset."""
    if dataset == 'lip':
        return LIP_LABELS
    elif dataset == 'atr':
        return ATR_LABELS
    elif dataset == 'pascal':
        return PASCAL_LABELS
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def extract_clothing_item(original_img, mask, class_indices, background='transparent'):
    """
    Extract clothing items from the original image using the parsing mask.
    
    Args:
        original_img: PIL Image or numpy array of the original image
        mask: numpy array of the parsing mask (pixel values are class indices)
        class_indices: list of class indices to extract (e.g., [5] for Upper-clothes, [5, 7] for Upper-clothes and Coat)
        background: 'transparent', 'white', or 'black' for the background
    
    Returns:
        PIL Image with extracted clothing items
    """
    if isinstance(original_img, Image.Image):
        original_img = np.array(original_img)
    
    # Create binary mask for selected classes
    binary_mask = np.zeros(mask.shape, dtype=np.uint8)
    for class_idx in class_indices:
        binary_mask[mask == class_idx] = 255
    
    # Create 3-channel mask
    mask_3channel = np.stack([binary_mask, binary_mask, binary_mask], axis=2) / 255.0
    
    # Extract clothing items
    if background == 'transparent':
        # Create RGBA image with transparency
        result = original_img.copy()
        alpha = (binary_mask > 0).astype(np.uint8) * 255
        result = np.dstack([result, alpha])
        result = Image.fromarray(result, 'RGBA')
    elif background == 'white':
        result = original_img.copy()
        result[binary_mask == 0] = 255
        result = Image.fromarray(result)
    elif background == 'black':
        result = original_img.copy()
        result[binary_mask == 0] = 0
        result = Image.fromarray(result)
    else:
        # Use mask to blend
        result = (original_img * mask_3channel).astype(np.uint8)
        result = Image.fromarray(result)
    
    return result


def list_available_items(dataset):
    """List all available clothing items for the dataset."""
    labels = get_label_mapping(dataset)
    print(f"\nAvailable items for {dataset.upper()} dataset:")
    print("-" * 50)
    for idx, label in enumerate(labels):
        print(f"  {idx:2d}: {label}")
    print("-" * 50)


def main():
    parser = argparse.ArgumentParser(description="Extract clothing pieces from human parsing results")
    parser.add_argument("--original-image", type=str, help="Path to original image")
    parser.add_argument("--parsing-mask", type=str, help="Path to parsing mask PNG")
    parser.add_argument("--dataset", type=str, default='lip', choices=['lip', 'atr', 'pascal'],
                       help="Dataset type (default: lip)")
    parser.add_argument("--items", type=str, nargs='+',
                       help="Clothing items to extract (e.g., 'Upper-clothes' 'Pants' or '5' '9' for indices)")
    parser.add_argument("--output", type=str, help="Output path for extracted image")
    parser.add_argument("--background", type=str, default='transparent', 
                       choices=['transparent', 'white', 'black', 'original'],
                       help="Background style (default: transparent)")
    parser.add_argument("--list-items", action='store_true', help="List all available items and exit")
    
    args = parser.parse_args()
    
    if args.list_items:
        list_available_items(args.dataset)
        return
    
    # Validate required arguments
    if not args.original_image or not args.parsing_mask or not args.items or not args.output:
        parser.error("--original-image, --parsing-mask, --items, and --output are required (unless using --list-items)")
    
    # Get label mapping
    labels = get_label_mapping(args.dataset)
    
    # Load images
    original_img = Image.open(args.original_image).convert('RGB')
    mask_img = Image.open(args.parsing_mask)
    
    # Convert mask to numpy array
    # If mask has palette (indexed color), pixel values are class indices (0-19)
    if mask_img.mode == 'P':
        # Indexed color mode - pixel values are already class indices
        mask = np.array(mask_img)
    else:
        # RGB or other mode - convert to grayscale
        mask = np.array(mask_img.convert('L'))
        # If values are too high, it might be a color visualization
        if mask.max() > len(labels):
            print("Warning: Mask appears to be a color visualization. Pixel values may not match class indices.")
            print(f"Max value in mask: {mask.max()}, expected max: {len(labels)-1}")
    
    # Parse items (can be names or indices)
    class_indices = []
    for item in args.items:
        try:
            # Try as index first
            idx = int(item)
            if 0 <= idx < len(labels):
                class_indices.append(idx)
                print(f"Extracting: {labels[idx]} (index {idx})")
            else:
                print(f"Warning: Index {idx} out of range. Skipping.")
        except ValueError:
            # Try as label name
            try:
                idx = labels.index(item)
                class_indices.append(idx)
                print(f"Extracting: {item} (index {idx})")
            except ValueError:
                print(f"Warning: '{item}' not found in labels. Skipping.")
    
    if not class_indices:
        print("Error: No valid items to extract!")
        return
    
    # Extract clothing items
    result = extract_clothing_item(original_img, mask, class_indices, background=args.background)
    
    # Save result
    result.save(args.output)
    print(f"\nExtracted image saved to: {args.output}")
    
    # Print summary
    item_names = [labels[idx] for idx in class_indices]
    print(f"Extracted items: {', '.join(item_names)}")


if __name__ == '__main__':
    main()

