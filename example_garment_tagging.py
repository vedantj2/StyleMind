#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Example usage of generate_garment_tags function.

This script demonstrates how to use the garment tagging feature
to extract structured metadata from reconstructed garment images.
"""

import json
import os
import sys
from pathlib import Path

# Add parent directory to path to import from app.py
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
from app import generate_garment_tags

# Load environment variables
load_dotenv()


def main():
    """Example usage of generate_garment_tags."""
    
    # Example: Tag a reconstructed t-shirt image
    image_path = "reconstructed_tshirt.png"
    
    # Check if image exists (or use a different path)
    if not Path(image_path).exists():
        print(f"⚠️  Image not found: {image_path}")
        print("Please provide a valid path to a reconstructed garment image.")
        print("\nExample usage:")
        print("  python example_garment_tagging.py path/to/your/image.png")
        
        # Try to find any reconstructed images in output directory
        output_dir = Path("output")
        if output_dir.exists():
            reconstructed_files = list(output_dir.glob("*_reconstructed.png"))
            if reconstructed_files:
                image_path = str(reconstructed_files[0])
                print(f"\nUsing found image: {image_path}")
            else:
                return
        else:
            return
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ ERROR: OPENAI_API_KEY not set in environment")
        print("Please set it in .env file or export it:")
        print("  export OPENAI_API_KEY='your-key-here'")
        return
    
    try:
        print(f"📸 Analyzing garment image: {image_path}")
        print("⏳ Generating tags...\n")
        
        # Generate tags
        tags = generate_garment_tags(image_path)
        
        # Print formatted JSON output
        print("✅ Successfully generated garment tags:\n")
        print(json.dumps(tags, indent=2, ensure_ascii=False))
        
        print("\n" + "="*60)
        print("Tag Summary:")
        print("="*60)
        print(f"Garment Type: {tags['garment_type']}")
        print(f"Sub Category: {tags['sub_category']}")
        print(f"Primary Color: {tags['primary_color']}")
        print(f"Secondary Colors: {', '.join(tags['secondary_colors']) if tags['secondary_colors'] else 'None'}")
        print(f"Fabric: {tags['fabric']}")
        print(f"Texture: {tags['texture']}")
        print(f"Pattern: {tags['pattern']}")
        print(f"Sleeve Type: {tags['sleeve_type']}")
        print(f"Fit: {tags['fit']}")
        print(f"Style: {tags['style']}")
        print(f"Season: {tags['season']}")
        print(f"Gender: {tags['gender']}")
        print(f"Keywords: {', '.join(tags['keywords'])}")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
    except ValueError as e:
        print(f"❌ Validation Error: {e}")
    except RuntimeError as e:
        print(f"❌ Runtime Error: {e}")
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")


if __name__ == "__main__":
    # Allow command-line argument for image path
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        # Override the default path
        import sys
        sys.modules[__name__].image_path = image_path
    
    main()

