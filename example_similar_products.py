#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Example usage of find_similar_products function.

This script demonstrates how to use the similar product discovery feature
to find visually and semantically similar clothing products online.
"""

import json
import os
import sys
from pathlib import Path

# Add parent directory to path to import from app.py
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
from app import find_similar_products

# Load environment variables
load_dotenv()


def main():
    """Example usage of find_similar_products."""
    
    # Example garment tags (from generate_garment_tags output)
    garment_tags = {
        "garment_type": "T-Shirt",
        "sub_category": "Crew Neck",
        "primary_color": "navy blue",
        "secondary_colors": [],
        "fabric": "cotton",
        "texture": "smooth",
        "pattern": "solid",
        "sleeve_type": "short sleeve",
        "fit": "regular",
        "style": "casual",
        "season": "all-season",
        "gender": "unisex",
        "keywords": ["casual", "comfortable", "basic"]
    }
    
    # Example image path (optional - not used for text similarity but reserved for future image similarity)
    image_path = "reconstructed_tshirt.png"
    
    # Check for API keys
    if not os.getenv("SERPAPI_API_KEY"):
        print("❌ ERROR: SERPAPI_API_KEY not set in environment")
        print("Please set it in .env file or export it:")
        print("  export SERPAPI_API_KEY='your-key-here'")
        print("\nGet your API key from: https://serpapi.com/")
        return
    
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ ERROR: OPENAI_API_KEY not set in environment")
        print("Please set it in .env file or export it:")
        print("  export OPENAI_API_KEY='your-key-here'")
        return
    
    try:
        print("🔍 Finding similar products...")
        print(f"📋 Garment: {garment_tags['garment_type']} - {garment_tags['primary_color']}")
        print(f"🏷️  Style: {garment_tags['style']}, Fabric: {garment_tags['fabric']}\n")
        
        # Find similar products
        results = find_similar_products(image_path, garment_tags)
        
        # Print results
        print(f"✅ Search Query Used: '{results['query_used']}'")
        print(f"📦 Found {len(results['results'])} similar products:\n")
        print("=" * 80)
        
        for idx, product in enumerate(results['results'], 1):
            print(f"\n{idx}. {product['title']}")
            print(f"   Store: {product['store']}")
            if product['price']:
                print(f"   Price: {product['currency']} ${product['price']:.2f}")
            else:
                print(f"   Price: Not available")
            print(f"   Similarity Score: {product['similarity_score']:.3f}")
            print(f"   URL: {product['url']}")
            if product['image']:
                print(f"   Image: {product['image']}")
        
        print("\n" + "=" * 80)
        
        # Save to JSON file
        output_file = "similar_products_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Results saved to: {output_file}")
        
    except ValueError as e:
        print(f"❌ Validation Error: {e}")
    except RuntimeError as e:
        print(f"❌ Runtime Error: {e}")
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")


if __name__ == "__main__":
    main()




