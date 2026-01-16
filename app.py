#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Flask API for Clothing Reconstruction Pipeline:
1) Accept image upload
2) Run parsing to get mask (Stage 1: SCHP)
3) Extract clothing items (Stage 1: Extraction)
4) Clean garment isolation (Stage 2: OpenCV cleanup)
5) Reconstruct with OpenAI API (Stage 3: gpt-image-1)

Requirements:
- OPENAI_API_KEY must be set in .env file or environment variables
- The checkpoint must exist at checkpoints/exp-schp-201908261155-lip.pth
"""

import base64
import io
import json
import logging
import os
import subprocess
import tempfile
import zipfile
from pathlib import Path
from difflib import SequenceMatcher
from typing import Dict, List, Optional

import cv2
import numpy as np
from dotenv import load_dotenv
from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from PIL import Image
from openai import OpenAI
from serpapi import GoogleSearch

# Load environment variables
load_dotenv()

# Configuration
MODEL_RESTORE = "checkpoints/exp-schp-201908261155-lip.pth"
DATASET = "lip"

# Clothing items to extract and reconstruct (LIP dataset indices)
CLOTHING_ITEMS = {
    "Hat": 1,
    "Glove": 3,
    "Sunglasses": 4,
    "Upper-clothes": 5,
    "Dress": 6,
    "Coat": 7,
    "Socks": 8,
    "Pants": 9,
    "Jumpsuits": 10,
    "Scarf": 11,
    "Left-shoe": 18,
    "Right-shoe": 19,
}

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def run_cmd(cmd, cwd):
    """Run a command and raise if it fails."""
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(cmd)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return result


def run_parsing(repo_root: Path, images_dir: Path, output_dir: Path):
    """Stage 1: Run SCHP parsing to generate mask."""
    simple_extractor_path = repo_root / "simple_extractor.py"
    if not simple_extractor_path.exists():
        raise FileNotFoundError(f"simple_extractor.py not found at {simple_extractor_path}")
    
    logging.info(f"Stage 1: Running parsing on input dir: {images_dir}")
    cmd = [
        "python",
        str(simple_extractor_path),
        "--dataset", DATASET,
        "--model-restore", str(repo_root / MODEL_RESTORE),
        "--input-dir", str(images_dir),
        "--output-dir", str(output_dir),
    ]
    run_cmd(cmd, cwd=repo_root)
    logging.info("✓ Parsing complete")


def extract_all_clothing_items(repo_root: Path, original_path: Path, mask_path: Path, output_dir: Path):
    """Stage 1 (continued): Extract all clothing items."""
    extract_all_path = repo_root / "extract_all_clothing.py"
    if not extract_all_path.exists():
        raise FileNotFoundError(f"extract_all_clothing.py not found at {extract_all_path}")
    
    logging.info(f"Stage 1: Extracting all clothing items from {original_path.name}")
    cmd = [
        "python",
        str(extract_all_path),
        "--original-image", str(original_path),
        "--parsing-mask", str(mask_path),
        "--output-dir", str(output_dir),
        "--dataset", DATASET,
        "--background", "transparent",
    ]
    run_cmd(cmd, cwd=repo_root)
    logging.info("✓ Extracted all clothing items")


def clean_garment_isolation(image_path: Path) -> Path:
    """
    Stage 2: Clean garment isolation using OpenCV.
    - Remove black background
    - Crop tightly to garment
    - Optional: morphological operations for cleanup
    """
    logging.info("Stage 2: Cleaning garment isolation...")
    
    # Read image with alpha channel
    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Handle different image formats
    if img.shape[2] == 4:
        # RGBA image
        alpha = img[:, :, 3]
        rgb = img[:, :, :3]
    elif img.shape[2] == 3:
        # RGB image - create alpha from non-black pixels
        rgb = img
        # Create alpha channel: non-black pixels are visible
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        alpha = (gray > 10).astype(np.uint8) * 255
    else:
        raise ValueError(f"Unsupported image format: {img.shape}")
    
    # Create mask from alpha channel
    mask = alpha > 0
    
    if not np.any(mask):
        raise ValueError("No visible content found in image (empty mask)")
    
    # Find bounding box
    ys, xs = np.where(mask)
    if len(ys) == 0 or len(xs) == 0:
        raise ValueError("Could not determine bounding box")
    
    y_min, y_max = min(ys), max(ys)
    x_min, x_max = min(xs), max(xs)
    
    # Crop tightly
    cropped_rgb = rgb[y_min:y_max+1, x_min:x_max+1]
    cropped_alpha = alpha[y_min:y_max+1, x_min:x_max+1]
    
    # Optional: Morphological closing to fill small holes
    kernel = np.ones((3, 3), np.uint8)
    cropped_alpha_clean = cv2.morphologyEx(cropped_alpha, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # Combine back to RGBA
    cropped_rgba = np.dstack([cropped_rgb, cropped_alpha_clean])
    
    # Convert BGR to RGB for PIL
    if cropped_rgba.shape[2] == 4:
        cropped_rgb_final = cv2.cvtColor(cropped_rgba[:, :, :3], cv2.COLOR_BGR2RGB)
        cropped_rgba_final = np.dstack([cropped_rgb_final, cropped_rgba[:, :, 3]])
    else:
        cropped_rgba_final = cv2.cvtColor(cropped_rgba, cv2.COLOR_BGR2RGB)
    
    # Save cleaned image
    output_path = image_path.parent / f"{image_path.stem}_cleaned.png"
    cleaned_img = Image.fromarray(cropped_rgba_final, 'RGBA')
    cleaned_img.save(output_path, "PNG")
    
    logging.info(f"✓ Cleaned garment saved: {output_path.name}")
    logging.info(f"  Original size: {img.shape[:2]}, Cropped size: {cropped_rgba_final.shape[:2]}")
    
    return output_path


def _generate_reconstruction_prompt(item_name: str = None) -> str:
    """
    Generate a custom reconstruction prompt based on the clothing item type.
    
    Args:
        item_name: Name of the clothing item from CLOTHING_ITEMS
    
    Returns:
        str: Item-specific reconstruction prompt
    """
    if not item_name:
        item_name = ""
    
    item_lower = item_name.lower()
    
    # Base prompt elements common to all items
    base_elements = [
        "Reconstruct this item as a high-quality ecommerce product image",
        "Preserve all colors, patterns, logos, and text exactly",
        "Centered composition",
        "Pure white background (RGB 255, 255, 255)",
        "No mannequin, no human body, no shadows",
        "Professional product photography style"
    ]
    
    # Item-specific instructions
    if "sunglasses" in item_lower or item_name == "Sunglasses":
        item_specific = [
            "This is a pair of sunglasses - reconstruct ONLY the sunglasses",
            "Show the sunglasses from a front-facing angle",
            "Lay them flat or show them in a natural resting position",
            "Preserve frame shape, lens color, and all design details",
            "Ensure both lenses are visible and properly aligned",
            "Maintain the bridge and temple arms structure"
        ]
    elif "shoe" in item_lower or "left-shoe" in item_lower or "right-shoe" in item_lower:
        item_specific = [
            "This is a shoe - reconstruct ONLY the shoe",
            "Show the shoe from a side profile or 3/4 angle",
            "Lay it flat on its side or sole",
            "Preserve shoe shape, sole design, laces, and all details",
            "Maintain proper proportions and structure",
            "Show the complete shoe including toe, heel, and sole"
        ]
    elif "hat" in item_lower or item_name == "Hat":
        item_specific = [
            "This is a hat - reconstruct ONLY the hat",
            "Show the hat from a front or side angle",
            "Lay it flat or show it in a natural position",
            "Preserve crown shape, brim, and all design elements",
            "Maintain proper hat proportions",
            "Show the complete hat structure"
        ]
    elif "glove" in item_lower or item_name == "Glove":
        item_specific = [
            "This is a glove - reconstruct ONLY the glove",
            "Show the glove from a front-facing angle",
            "Lay it flat with fingers spread naturally",
            "Preserve finger structure, palm, and all details",
            "Maintain proper glove proportions",
            "Show the complete glove including all fingers"
        ]
    elif "sock" in item_lower or item_name == "Socks":
        item_specific = [
            "This is a sock - reconstruct ONLY the sock",
            "Show the sock from a side or front angle",
            "Lay it flat or show it in a natural position",
            "Preserve sock shape, cuff, and all design patterns",
            "Maintain proper sock proportions",
            "Show the complete sock structure"
        ]
    elif "pants" in item_lower or item_name == "Pants":
        item_specific = [
            "This is a pair of pants - reconstruct ONLY the pants",
            "Lay them completely flat",
            "Show the pants from the front view",
            "Symmetric shape with both legs aligned",
            "Remove all wrinkles and folds",
            "Preserve fabric texture, color, and all details",
            "Show waistband, zipper/fly, and leg openings clearly"
        ]
    elif "dress" in item_lower or item_name == "Dress":
        item_specific = [
            "This is a dress - reconstruct ONLY the dress",
            "Lay it completely flat",
            "Show the dress from the front view",
            "Symmetric shape",
            "Remove all wrinkles and folds",
            "Preserve fabric texture, color, patterns, and all design details",
            "Show neckline, sleeves (if any), and hem clearly"
        ]
    elif "coat" in item_lower or item_name == "Coat":
        item_specific = [
            "This is a coat - reconstruct ONLY the coat",
            "Lay it completely flat",
            "Show the coat from the front view",
            "Symmetric shape",
            "Remove all wrinkles and folds",
            "Preserve fabric texture, color, and all details",
            "Show collar, buttons/zipper, and sleeves clearly"
        ]
    elif "jumpsuit" in item_lower or item_name == "Jumpsuits":
        item_specific = [
            "This is a jumpsuit - reconstruct ONLY the jumpsuit",
            "Lay it completely flat",
            "Show the jumpsuit from the front view",
            "Symmetric shape",
            "Remove all wrinkles and folds",
            "Preserve fabric texture, color, and all details",
            "Show top portion, waist, and leg portions clearly"
        ]
    elif "scarf" in item_lower or item_name == "Scarf":
        item_specific = [
            "This is a scarf - reconstruct ONLY the scarf",
            "Lay it completely flat",
            "Show the scarf in a natural draped or folded position",
            "Preserve fabric texture, color, patterns, and all details",
            "Show the full length and width of the scarf",
            "Maintain proper scarf proportions"
        ]
    elif "upper" in item_lower or item_name == "Upper-clothes":
        item_specific = [
            "This is an upper body garment (shirt, t-shirt, etc.) - reconstruct ONLY this garment",
            "Lay it completely flat",
            "Show the garment from the front view",
            "Symmetric shape",
            "Remove all wrinkles and folds",
            "Preserve fabric texture exactly",
            "Preserve all colors, patterns, logos, and text exactly",
            "Show neckline, sleeves, and hem clearly"
        ]
    else:
        # Generic prompt for unknown items
        item_specific = [
            f"This is a {item_name or 'clothing item'} - reconstruct ONLY this item",
            "Lay it completely flat",
            "Symmetric shape",
            "Remove all wrinkles and folds",
            "Preserve fabric texture exactly",
            "Preserve all colors, patterns, logos, and text exactly"
        ]
    
    # Combine base and item-specific elements
    prompt_lines = base_elements + item_specific
    
    # Format as a clean prompt
    prompt = "\n".join(f"    - {line}" for line in prompt_lines)
    prompt = prompt.strip()
    
    return prompt


def reconstruct_with_openai(image_path: Path, item_name: str = None) -> Path:
    """
    Stage 3: Reconstruct clothing using OpenAI Image API.
    Uses gpt-image-1 model for image-to-image reconstruction.
    
    Args:
        image_path: Path to the cleaned garment image
        item_name: Name of the clothing item (e.g., "Sunglasses", "Left-shoe", "Upper-clothes")
                   Used to generate item-specific reconstruction prompts.
    """
    logging.info(f"Stage 3: Reconstructing {item_name or 'garment'} with OpenAI API...")
    
    # Generate item-specific prompt
    prompt = _generate_reconstruction_prompt(item_name)
    
    try:
        # Use the image edit endpoint with gpt-image-1 for image-to-image
        # GPT Image models support the Edits endpoint for modifying existing images
        with open(image_path, "rb") as image_file:
            result = client.images.edit(
                model="gpt-image-1",
                image=image_file,
                prompt=prompt.strip(),
                size="1024x1024",
                n=1
            )
        
        # Handle response
        if hasattr(result.data[0], 'b64_json'):
            # Base64 encoded response
            image_base64 = result.data[0].b64_json
            image_bytes = base64.b64decode(image_base64)
        elif hasattr(result.data[0], 'url'):
            # URL response
            import requests
            image_url = result.data[0].url
            image_response = requests.get(image_url)
            image_response.raise_for_status()
            image_bytes = image_response.content
        else:
            raise ValueError("Unexpected response format from OpenAI API")
        
        # Save reconstructed image
        output_path = image_path.parent / f"{image_path.stem}_reconstructed.png"
        with open(output_path, "wb") as f:
            f.write(image_bytes)
        
        logging.info(f"✓ Reconstructed image saved: {output_path.name}")
        return output_path
        
    except Exception as e:
        # Fallback to DALL-E 3 if gpt-image-1 edit fails
        logging.warning(f"gpt-image-1 edit failed, falling back to DALL-E 3: {e}")
        return reconstruct_with_dalle3_fallback(image_path, prompt)


def reconstruct_with_dalle3_fallback(image_path: Path, prompt: str) -> Path:
    """Fallback: Use DALL-E 3 text-to-image (requires image description)."""
    logging.info("Using DALL-E 3 fallback (text-to-image)...")
    
    # First, get image description using Vision API
    with open(image_path, "rb") as f:
        image_data = f.read()
    image_b64 = base64.b64encode(image_data).decode()
    
    vision_response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Describe this upper body clothing item in extreme detail - colors, patterns, design, text, logos, brand names, everything visible. Be very specific."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_b64}"
                        }
                    }
                ]
            }
        ],
        max_tokens=500
    )
    
    clothing_description = vision_response.choices[0].message.content
    logging.info(f"Vision description: {clothing_description[:200]}...")
    
    # Combine description with reconstruction prompt
    full_prompt = f"{clothing_description}. {prompt}"
    
    # Generate with DALL-E 3
    result = client.images.generate(
        model="dall-e-3",
        prompt=full_prompt,
        size="1024x1024",
        quality="hd",
        n=1
    )
    
    # Download image
    import requests
    image_url = result.data[0].url
    image_response = requests.get(image_url)
    image_response.raise_for_status()
    
    output_path = image_path.parent / f"{image_path.stem}_reconstructed.png"
    with open(output_path, "wb") as f:
        f.write(image_response.content)
    
    logging.info(f"✓ Reconstructed image saved (DALL-E 3): {output_path.name}")
    return output_path


def generate_garment_tags(image_path: str) -> dict:
    """
    Generate structured ecommerce tags from a reconstructed garment image.
    
    Uses OpenAI's multimodal model (GPT-4o) to analyze the garment and extract
    structured metadata suitable for database storage.
    
    Args:
        image_path: Path to the garment image file (PNG, JPG, etc.)
    
    Returns:
        dict: Structured JSON object with garment tags:
            {
                "garment_type": string,
                "sub_category": string,
                "primary_color": string,
                "secondary_colors": string[],
                "fabric": string,
                "texture": string,
                "pattern": string,
                "sleeve_type": string,
                "fit": string,
                "style": string,
                "season": string,
                "gender": "men" | "women" | "unisex",
                "keywords": string[]
            }
    
    Raises:
        FileNotFoundError: If image_path does not exist
        ValueError: If JSON parsing fails or response is invalid
        RuntimeError: If OpenAI API call fails
    
    Example:
        >>> tags = generate_garment_tags("reconstructed_tshirt.png")
        >>> print(json.dumps(tags, indent=2))
    """
    image_path_obj = Path(image_path)
    if not image_path_obj.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Read and encode image as base64
    with open(image_path_obj, "rb") as f:
        image_data = f.read()
    image_b64 = base64.b64encode(image_data).decode()
    
    # Determine MIME type from file extension
    ext = image_path_obj.suffix.lower()
    if ext in ['.jpg', '.jpeg']:
        mime_type = "image/jpeg"
    elif ext == '.png':
        mime_type = "image/png"
    elif ext == '.gif':
        mime_type = "image/gif"
    elif ext == '.webp':
        mime_type = "image/webp"
    else:
        mime_type = "image/png"  # Default
    
    # Strict JSON prompt for garment tagging
    prompt = """Analyze ONLY the garment in this image (ignore background, mannequin, or any non-garment elements).

Return STRICT JSON only. No markdown, no explanations, no code blocks. Just valid JSON.

The JSON must have exactly this structure:
{
  "garment_type": "string (e.g., T-Shirt, Dress, Pants, Jacket)",
  "sub_category": "string (e.g., T-Shirt -> Crew Neck, V-Neck, Henley)",
  "primary_color": "string (e.g., navy blue, white, black)",
  "secondary_colors": ["string array of additional colors if any"],
  "fabric": "string (best guess: cotton, polyester, denim, silk, wool, etc.)",
  "texture": "string (best guess: smooth, ribbed, textured, knit, etc.)",
  "pattern": "string (solid, striped, floral, geometric, plaid, etc. or 'none' if solid)",
  "sleeve_type": "string (long sleeve, short sleeve, sleeveless, cap sleeve, etc.)",
  "fit": "string (slim, regular, loose, oversized, fitted, etc.)",
  "style": "string (casual, formal, sporty, vintage, modern, etc.)",
  "season": "string (spring, summer, fall, winter, all-season)",
  "gender": "men" | "women" | "unisex",
  "keywords": ["array of relevant descriptive keywords"]
}

Important:
- Return ONLY the JSON object, nothing else
- All string values must be non-empty
- secondary_colors and keywords can be empty arrays [] if none apply
- Use best-guess inference for fabric and texture if uncertain
- Analyze only the garment, not background or other elements"""

    try:
        logging.info(f"Generating tags for: {image_path_obj.name}")
        
        # Call OpenAI Vision API with GPT-4o (latest multimodal model)
        # Note: GPT-4.1 doesn't exist; using GPT-4o which is the latest and best for vision
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{image_b64}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=1000,
            temperature=0.3  # Lower temperature for more consistent structured output
        )
        
        raw_content = response.choices[0].message.content.strip()
        logging.debug(f"Raw model output: {raw_content[:200]}...")
        
        # Clean the response - remove markdown code blocks if present
        if raw_content.startswith("```json"):
            raw_content = raw_content[7:]  # Remove ```json
        elif raw_content.startswith("```"):
            raw_content = raw_content[3:]  # Remove ```
        
        if raw_content.endswith("```"):
            raw_content = raw_content[:-3]  # Remove trailing ```
        
        raw_content = raw_content.strip()
        
        # Parse JSON
        try:
            tags = json.loads(raw_content)
        except json.JSONDecodeError as e:
            logging.error(f"JSON parsing failed. Raw output: {raw_content}")
            raise ValueError(
                f"Failed to parse JSON response from OpenAI API. "
                f"JSON decode error: {e}. "
                f"Raw output (first 500 chars): {raw_content[:500]}"
            ) from e
        
        # Validate required fields
        required_fields = [
            "garment_type", "sub_category", "primary_color", "secondary_colors",
            "fabric", "texture", "pattern", "sleeve_type", "fit", "style",
            "season", "gender", "keywords"
        ]
        
        missing_fields = [field for field in required_fields if field not in tags]
        if missing_fields:
            raise ValueError(
                f"Response missing required fields: {missing_fields}. "
                f"Received: {list(tags.keys())}"
            )
        
        # Validate gender field
        if tags["gender"] not in ["men", "women", "unisex"]:
            logging.warning(
                f"Invalid gender value: {tags['gender']}. Expected 'men', 'women', or 'unisex'. "
                f"Defaulting to 'unisex'."
            )
            tags["gender"] = "unisex"
        
        # Ensure arrays are lists
        if not isinstance(tags["secondary_colors"], list):
            tags["secondary_colors"] = []
        if not isinstance(tags["keywords"], list):
            tags["keywords"] = []
        
        logging.info(f"✓ Successfully generated tags for {image_path_obj.name}")
        return tags
        
    except FileNotFoundError:
        raise
    except ValueError:
        raise
    except Exception as e:
        logging.error(f"OpenAI API call failed: {e}")
        raise RuntimeError(f"Failed to generate garment tags: {e}") from e


def find_similar_products(garment_image_path: str, garment_tags: dict) -> dict:
    """
    Find visually and semantically similar clothing products online.
    
    Uses GPT-4o to generate an optimized search query from garment tags,
    then searches for similar products using SerpAPI, ranks them by similarity,
    and returns structured product information with prices.
    
    Args:
        garment_image_path: Path to the reconstructed garment image (for future image similarity)
        garment_tags: Dictionary containing garment metadata from generate_garment_tags()
            Expected keys: garment_type, sub_category, primary_color, fabric, style, etc.
    
    Returns:
        dict: Structured product discovery results:
            {
                "query_used": string,
                "results": [
                    {
                        "title": string,
                        "price": number,
                        "currency": string,
                        "store": string,
                        "url": string,
                        "image": string,
                        "similarity_score": number (0-1)
                    }
                ]
            }
    
    Raises:
        ValueError: If garment_tags is invalid or missing required fields
        RuntimeError: If API calls fail or no results found
    
    Example:
        >>> tags = {
        ...     "garment_type": "T-Shirt",
        ...     "sub_category": "Crew Neck",
        ...     "primary_color": "navy blue",
        ...     "fabric": "cotton",
        ...     "style": "casual"
        ... }
        >>> results = find_similar_products("reconstructed_tshirt.png", tags)
        >>> print(json.dumps(results, indent=2))
    """
    # Validate garment_tags
    required_fields = ["garment_type", "primary_color"]
    missing_fields = [field for field in required_fields if field not in garment_tags]
    if missing_fields:
        raise ValueError(f"Missing required fields in garment_tags: {missing_fields}")
    
    # Validate API keys
    serpapi_key = os.getenv("SERPAPI_API_KEY")
    if not serpapi_key:
        raise RuntimeError("SERPAPI_API_KEY not set in environment")
    
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set in environment")
    
    try:
        # Step 1: Generate optimized search query from tags using GPT-4o
        logging.info("Generating search query from garment tags...")
        
        query_prompt = f"""Generate a concise, effective Google Shopping search query for finding similar products to this garment.

Garment Details:
- Type: {garment_tags.get('garment_type', 'N/A')}
- Sub Category: {garment_tags.get('sub_category', 'N/A')}
- Primary Color: {garment_tags.get('primary_color', 'N/A')}
- Secondary Colors: {', '.join(garment_tags.get('secondary_colors', [])) if garment_tags.get('secondary_colors') else 'None'}
- Fabric: {garment_tags.get('fabric', 'N/A')}
- Style: {garment_tags.get('style', 'N/A')}
- Pattern: {garment_tags.get('pattern', 'N/A')}
- Gender: {garment_tags.get('gender', 'N/A')}
- Keywords: {', '.join(garment_tags.get('keywords', [])) if garment_tags.get('keywords') else 'None'}

Requirements:
- Return ONLY the search query text, nothing else
- Keep it concise (5-10 words max)
- Focus on the most distinctive features (type, color, style)
- Use natural language that shoppers would use
- Do NOT include price ranges or brand names
- Example format: "navy blue cotton crew neck t-shirt"

Search Query:"""
        
        query_response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": query_prompt
                }
            ],
            max_tokens=50,
            temperature=0.3
        )
        
        search_query = query_response.choices[0].message.content.strip()
        # Clean up any quotes or extra formatting
        search_query = search_query.strip('"').strip("'").strip()
        logging.info(f"Generated search query: {search_query}")
        
        # Step 2: Search for products using SerpAPI
        logging.info("Searching for similar products...")
        
        params = {
            "engine": "google_shopping",
            "q": search_query,
            "api_key": serpapi_key,
            "num": 20,  # Get more results to filter and rank
            "tbs": "vw:g"  # Google Shopping filter
        }
        
        search = GoogleSearch(params)
        results = search.get_dict()
        
        # Extract shopping results
        shopping_results = results.get("shopping_results", [])
        if not shopping_results:
            logging.warning("No shopping results found")
            return {
                "query_used": search_query,
                "results": []
            }
        
        # Step 3: Process and rank results
        processed_results = []
        seen_stores = set()
        
        for item in shopping_results:
            # Skip if we've already seen this store (remove duplicates)
            store = item.get("source", "").lower()
            if store in seen_stores:
                continue
            
            # Extract product information
            title = item.get("title", "")
            price_str = item.get("price", "")
            url = item.get("link", "")
            image = item.get("thumbnail", "")
            source = item.get("source", "")
            
            # Parse price
            price = None
            currency = "USD"
            if price_str:
                # Extract numeric value and currency
                import re
                price_match = re.search(r'([\d,]+\.?\d*)', price_str.replace(',', ''))
                if price_match:
                    try:
                        price = float(price_match.group(1))
                    except ValueError:
                        pass
                
                # Detect currency
                if '$' in price_str or 'USD' in price_str.upper():
                    currency = "USD"
                elif '€' in price_str or 'EUR' in price_str.upper():
                    currency = "EUR"
                elif '£' in price_str or 'GBP' in price_str.upper():
                    currency = "GBP"
            
            # Skip if essential data is missing
            if not title or not url:
                continue
            
            # Compute similarity score
            similarity_score = _compute_similarity_score(garment_tags, title)
            
            processed_results.append({
                "title": title,
                "price": price,
                "currency": currency,
                "store": source,
                "url": url,
                "image": image,
                "similarity_score": round(similarity_score, 3)
            })
            
            seen_stores.add(store)
        
        # Step 4: Sort by similarity score (descending) and limit to top 10
        processed_results.sort(key=lambda x: x["similarity_score"], reverse=True)
        top_results = processed_results[:10]
        
        logging.info(f"Found {len(top_results)} similar products")
        
        return {
            "query_used": search_query,
            "results": top_results
        }
        
    except Exception as e:
        logging.error(f"Failed to find similar products: {e}")
        raise RuntimeError(f"Failed to find similar products: {e}") from e


def _compute_similarity_score(garment_tags: dict, product_title: str) -> float:
    """
    Compute similarity score between garment tags and product title.
    
    Uses text similarity metrics to compare key attributes.
    
    Args:
        garment_tags: Dictionary of garment metadata
        product_title: Product title from search results
    
    Returns:
        float: Similarity score between 0 and 1
    """
    if not product_title:
        return 0.0
    
    title_lower = product_title.lower()
    score = 0.0
    max_score = 0.0
    
    # Check garment type (highest weight)
    garment_type = garment_tags.get("garment_type", "").lower()
    if garment_type:
        max_score += 0.3
        if garment_type in title_lower:
            score += 0.3
        else:
            # Partial match using sequence matcher
            similarity = SequenceMatcher(None, garment_type, title_lower).ratio()
            score += similarity * 0.3
    
    # Check sub category
    sub_category = garment_tags.get("sub_category", "").lower()
    if sub_category:
        max_score += 0.2
        if sub_category in title_lower:
            score += 0.2
        else:
            similarity = SequenceMatcher(None, sub_category, title_lower).ratio()
            score += similarity * 0.2
    
    # Check primary color
    primary_color = garment_tags.get("primary_color", "").lower()
    if primary_color:
        max_score += 0.2
        # Split color into words (e.g., "navy blue" -> ["navy", "blue"])
        color_words = primary_color.split()
        for word in color_words:
            if word in title_lower:
                score += 0.2 / len(color_words)
                break
        else:
            # Partial match
            similarity = SequenceMatcher(None, primary_color, title_lower).ratio()
            score += similarity * 0.2
    
    # Check fabric
    fabric = garment_tags.get("fabric", "").lower()
    if fabric:
        max_score += 0.15
        if fabric in title_lower:
            score += 0.15
        else:
            similarity = SequenceMatcher(None, fabric, title_lower).ratio()
            score += similarity * 0.15
    
    # Check style
    style = garment_tags.get("style", "").lower()
    if style:
        max_score += 0.15
        if style in title_lower:
            score += 0.15
        else:
            similarity = SequenceMatcher(None, style, title_lower).ratio()
            score += similarity * 0.15
    
    # Normalize score to 0-1 range
    if max_score > 0:
        normalized_score = score / max_score
    else:
        normalized_score = 0.0
    
    return min(normalized_score, 1.0)


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "message": "Clothing reconstruction API is running"})


@app.route("/reconstruct", methods=["POST"])
def reconstruct():
    """
    Main endpoint: Full pipeline from image upload to reconstructed product images with tags.
    
    Extracts and reconstructs all clothing items found in the image:
    - Hat, Glove, Sunglasses, Upper-clothes, Dress, Coat
    - Socks, Pants, Jumpsuits, Scarf, Left-shoe, Right-shoe
    
    Automatically generates structured ecommerce tags for each reconstructed garment.
    
    Request:
        - Form data with 'file' field containing image
    
    Response:
        - JSON with base64 encoded images and tags:
          {
            "originalImage": "data:image/...;base64,...",
            "maskImage": "data:image/png;base64,...",
            "reconstructedImages": [
              {
                "name": "item_name",
                "image": "data:image/png;base64,...",
                "tags": {
                  "garment_type": string,
                  "sub_category": string,
                  "primary_color": string,
                  "secondary_colors": string[],
                  "fabric": string,
                  "texture": string,
                  "pattern": string,
                  "sleeve_type": string,
                  "fit": string,
                  "style": string,
                  "season": string,
                  "gender": "men" | "women" | "unisex",
                  "keywords": string[]
                }
              },
              ...
            ]
          }
        
        Note: If tagging fails for an item, the "tags" field will be omitted for that item.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded (form field 'file' missing)"}), 400
    
    file = request.files["file"]
    if not file or file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    
    # Validate API key
    if not os.getenv("OPENAI_API_KEY"):
        return jsonify({"error": "OPENAI_API_KEY not set in environment"}), 500
    
    repo_root = Path(__file__).resolve().parent
    model_path = repo_root / MODEL_RESTORE
    if not model_path.exists():
        return jsonify({"error": f"Checkpoint not found at {model_path}"}), 500
    
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            images_dir = tmpdir / "images"
            output_dir = tmpdir / "output"
            images_dir.mkdir(parents=True, exist_ok=True)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save uploaded image
            ext = Path(file.filename).suffix or ".png"
            input_path = images_dir / f"input{ext}"
            file.save(input_path)
            logging.info(f"✓ Uploaded image: {input_path.name}")
            
            # Stage 1: Run parsing
            run_parsing(repo_root, images_dir, output_dir)
            
            # Get mask path - simple_extractor saves as {basename}.png
            # Handle different extensions properly (e.g., input.jpeg -> input.png, not input.j.png)
            input_basename = input_path.stem  # Gets name without extension
            mask_path = output_dir / f"{input_basename}.png"
            
            # If mask doesn't exist, try to find any PNG file in output_dir (fallback)
            if not mask_path.exists():
                png_files = list(output_dir.glob("*.png"))
                if png_files:
                    mask_path = png_files[0]
                    logging.info(f"Using found mask: {mask_path.name}")
                else:
                    return jsonify({"error": f"Parsing mask not generated. Expected: {mask_path}"}), 500
            
            # Stage 1 (continued): Extract all clothing items
            clothes_dir = output_dir / "clothing"
            clothes_dir.mkdir(parents=True, exist_ok=True)
            extract_all_clothing_items(repo_root, input_path, mask_path, clothes_dir)
            
            # Find all extracted clothing items
            extracted_files = list(clothes_dir.glob("*.png"))
            if not extracted_files:
                return jsonify({"error": "No clothing items found or extracted"}), 500
            
            # Filter to only the items we want to reconstruct
            items_to_reconstruct = []
            for extracted_file in extracted_files:
                # Extract index from filename (format: 02_hair.png -> index 2)
                try:
                    idx_str = extracted_file.stem.split('_')[0]
                    idx = int(idx_str)
                    # Check if this index is in our CLOTHING_ITEMS
                    if idx in CLOTHING_ITEMS.values():
                        items_to_reconstruct.append(extracted_file)
                        item_name = [k for k, v in CLOTHING_ITEMS.items() if v == idx][0]
                        logging.info(f"Found item to reconstruct: {item_name} ({extracted_file.name})")
                except (ValueError, IndexError):
                    continue
            
            if not items_to_reconstruct:
                return jsonify({"error": "No matching clothing items found for reconstruction"}), 500
            
            # Process each item: clean, reconstruct, and tag
            reconstructed_images = []
            for item_path in items_to_reconstruct:
                # Extract item name from CLOTHING_ITEMS dictionary
                try:
                    idx_str = item_path.stem.split('_')[0]
                    idx = int(idx_str)
                    # Find the item name from CLOTHING_ITEMS
                    item_name = None
                    for name, item_idx in CLOTHING_ITEMS.items():
                        if item_idx == idx:
                            item_name = name
                            break
                    
                    # Fallback to filename if not found in dictionary
                    if not item_name:
                        item_name = item_path.stem.split('_', 1)[1] if '_' in item_path.stem else item_path.stem
                except (ValueError, IndexError):
                    item_name = item_path.stem.split('_', 1)[1] if '_' in item_path.stem else item_path.stem
                
                logging.info(f"Processing: {item_name}")
                
                try:
                    # Stage 2: Clean garment isolation
                    cleaned_path = clean_garment_isolation(item_path)
                    
                    # Stage 3: Reconstruct with OpenAI (pass item name for custom prompt)
                    reconstructed_path = reconstruct_with_openai(cleaned_path, item_name)
                    
                    # Stage 4: Generate tags for reconstructed garment
                    tags = None
                    try:
                        logging.info(f"Generating tags for: {item_name}")
                        tags = generate_garment_tags(str(reconstructed_path))
                        logging.info(f"✓ Tags generated for: {item_name}")
                    except Exception as tag_error:
                        logging.warning(f"Failed to generate tags for {item_name}: {tag_error}")
                        # Continue without tags if tagging fails
                    
                    reconstructed_images.append({
                        "name": item_name,
                        "path": reconstructed_path,
                        "tags": tags
                    })
                    logging.info(f"✓ Reconstructed: {item_name}")
                except Exception as e:
                    logging.warning(f"Failed to reconstruct {item_name}: {e}")
                    continue
            
            if not reconstructed_images:
                return jsonify({"error": "Failed to reconstruct any clothing items"}), 500
            
            # Convert images to base64 for JSON response
            # Original image - determine MIME type from extension
            mime_type = "image/png"
            if ext.lower() in ['.jpg', '.jpeg']:
                mime_type = "image/jpeg"
            elif ext.lower() == '.gif':
                mime_type = "image/gif"
            
            with open(input_path, "rb") as f:
                original_image_b64 = base64.b64encode(f.read()).decode('utf-8')
            
            # Mask image
            with open(mask_path, "rb") as f:
                mask_image_b64 = base64.b64encode(f.read()).decode('utf-8')
            
            # Reconstructed images with tags
            reconstructed_data = []
            for img_info in reconstructed_images:
                with open(img_info["path"], "rb") as f:
                    img_b64 = base64.b64encode(f.read()).decode('utf-8')
                    item_data = {
                        "name": img_info["name"],
                        "image": f"data:image/png;base64,{img_b64}"
                    }
                    # Include tags if available
                    if img_info.get("tags"):
                        item_data["tags"] = img_info["tags"]
                    reconstructed_data.append(item_data)
            
            logging.info(f"✓ Pipeline complete - returning {len(reconstructed_images)} reconstructed images")
            
            return jsonify({
                "originalImage": f"data:{mime_type};base64,{original_image_b64}",
                "maskImage": f"data:image/png;base64,{mask_image_b64}",
                "reconstructedImages": reconstructed_data
            })
    
    except Exception as e:
        logging.error(f"API Error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/tag-garment", methods=["POST"])
def tag_garment():
    """
    Endpoint: Generate structured ecommerce tags from a reconstructed garment image.
    
    Request:
        - Form data with 'file' field containing image, OR
        - JSON with 'image_path' field containing path to image on server
    
    Response:
        - JSON with structured garment tags:
          {
            "garment_type": string,
            "sub_category": string,
            "primary_color": string,
            "secondary_colors": string[],
            "fabric": string,
            "texture": string,
            "pattern": string,
            "sleeve_type": string,
            "fit": string,
            "style": string,
            "season": string,
            "gender": "men" | "women" | "unisex",
            "keywords": string[]
          }
    """
    # Validate API key
    if not os.getenv("OPENAI_API_KEY"):
        return jsonify({"error": "OPENAI_API_KEY not set in environment"}), 500
    
    try:
        # Check if image is uploaded via form data
        if "file" in request.files:
            file = request.files["file"]
            if not file or file.filename == "":
                return jsonify({"error": "No selected file"}), 400
            
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
                file.save(tmp_file.name)
                image_path = tmp_file.name
                temp_file_created = True
        
        # Check if image path is provided in JSON
        elif request.is_json and "image_path" in request.json:
            image_path = request.json["image_path"]
            temp_file_created = False
        
        else:
            return jsonify({
                "error": "No image provided. Send 'file' in form data or 'image_path' in JSON"
            }), 400
        
        # Generate tags
        tags = generate_garment_tags(image_path)
        
        # Clean up temporary file if created
        if temp_file_created:
            try:
                os.unlink(image_path)
            except Exception:
                pass
        
        return jsonify(tags)
    
    except FileNotFoundError as e:
        return jsonify({"error": f"Image not found: {str(e)}"}), 404
    except ValueError as e:
        return jsonify({"error": f"Validation error: {str(e)}"}), 400
    except RuntimeError as e:
        return jsonify({"error": f"API error: {str(e)}"}), 500
    except Exception as e:
        logging.error(f"Tagging Error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/find-similar-products", methods=["POST"])
def find_similar_products_endpoint():
    """
    Endpoint: Find similar products for a reconstructed garment.
    
    Request:
        - JSON with 'image_path' (optional) and 'tags' (required):
          {
            "image_path": "path/to/reconstructed_image.png",  // optional
            "tags": {
              "garment_type": "T-Shirt",
              "sub_category": "Crew Neck",
              ...
            }
          }
    
    Response:
        - JSON with similar products:
          {
            "query_used": string,
            "results": [...]
          }
    """
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    
    data = request.json
    
    # Validate tags
    if "tags" not in data:
        return jsonify({"error": "Missing 'tags' field in request"}), 400
    
    tags = data["tags"]
    image_path = data.get("image_path", "")
    
    # Validate API keys
    if not os.getenv("SERPAPI_API_KEY"):
        return jsonify({"error": "SERPAPI_API_KEY not set in environment"}), 500
    
    if not os.getenv("OPENAI_API_KEY"):
        return jsonify({"error": "OPENAI_API_KEY not set in environment"}), 500
    
    try:
        # If image_path is provided, verify it exists
        if image_path and not Path(image_path).exists():
            return jsonify({"error": f"Image not found: {image_path}"}), 404
        
        results = find_similar_products(image_path or "", tags)
        return jsonify(results)
    
    except ValueError as e:
        return jsonify({"error": f"Validation error: {str(e)}"}), 400
    except RuntimeError as e:
        return jsonify({"error": f"API error: {str(e)}"}), 500
    except Exception as e:
        logging.error(f"Similar products error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
