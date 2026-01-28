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
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
from dotenv import load_dotenv
from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from PIL import Image
from openai import OpenAI
import requests
from pymongo import MongoClient
from datetime import datetime, timezone
import uuid
import boto3
from botocore.exceptions import ClientError
from botocore.exceptions import ClientError

# Load environment variables
load_dotenv()

# Configuration
MODEL_RESTORE = "checkpoints/exp-schp-201908261155-lip.pth"
DATASET = "lip"

# AWS Lambda URL for presigned URL generation
LAMBDA_PRESIGNED_URL = "https://kjym4xf6sgvorxpst7dbbcf2ge0szprk.lambda-url.ap-south-1.on.aws/"

# S3 Bucket configuration
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "wardrobe-managment-vj")
S3_REGION = os.getenv("S3_REGION", "ap-south-1")

# Initialize boto3 S3 client for generating presigned URLs
try:
    s3_client = boto3.client('s3', region_name=S3_REGION)
    logging.info(f"✓ Initialized S3 client for region: {S3_REGION}")
except Exception as e:
    logging.warning(f"Failed to initialize S3 client: {e}. Presigned URL generation will fail.")
    s3_client = None

# MongoDB configuration
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb+srv://vedantjain0210_db_user:B3uuEkqgw8hV8oWa@wardrobe.eamzexq.mongodb.net/")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "TestDB")
MONGODB_COLLECTION_NAME = os.getenv("MONGODB_COLLECTION_NAME", "Clothing")
RECOMMENDED_OUTFITS_COLLECTION_NAME = os.getenv("RECOMMENDED_OUTFITS_COLLECTION_NAME", "recommended_outfits")

# Initialize MongoDB client
try:
    mongodb_client = MongoClient(MONGODB_URI)
    mongodb_db = mongodb_client[MONGODB_DB_NAME]
    mongodb_collection = mongodb_db[MONGODB_COLLECTION_NAME]
    outfits_collection = mongodb_db[RECOMMENDED_OUTFITS_COLLECTION_NAME]
    logging.info(f"✓ Connected to MongoDB: {MONGODB_DB_NAME}.{MONGODB_COLLECTION_NAME} and {RECOMMENDED_OUTFITS_COLLECTION_NAME}")
except Exception as e:
    logging.warning(f"Failed to connect to MongoDB: {e}. MongoDB operations will fail.")
    mongodb_client = None
    mongodb_db = None
    mongodb_collection = None
    outfits_collection = None

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

# ---------------------------
# Controlled vocabularies for normalization/enrichment
# ---------------------------
GARMENT_ROLE_ENUM = {"top", "bottom", "footwear", "outerwear", "accessory"}
SEASON_ENUM = {"spring", "summer", "fall", "autumn", "winter", "all-season"}
COLOR_TEMPERATURE_ENUM = {"warm", "cool", "neutral"}

# garment_type -> garment_role
GARMENT_ROLE_MAP = {
    "t-shirt": "top",
    "shirt": "top",
    "blouse": "top",
    "polo": "top",
    "tank": "top",
    "sweater": "top",
    "hoodie": "top",
    "dress": "top",
    "jacket": "outerwear",
    "coat": "outerwear",
    "scarf": "accessory",
    "gloves": "accessory",
    "hat": "accessory",
    "cap": "accessory",
    "beanie": "accessory",
    "belt": "accessory",
    "bag": "accessory",
    "skirt": "bottom",
    "jeans": "bottom",
    "pants": "bottom",
    "trousers": "bottom",
    "shorts": "bottom",
    "joggers": "bottom",
    "sweatpants": "bottom",
    "leggings": "bottom",
    "socks": "accessory",
    "shoe": "footwear",
    "shoes": "footwear",
    "boots": "footwear",
    "sneakers": "footwear",
    "sandals": "footwear",
    "flip-flops": "footwear",
}


def _crop_to_alpha_bbox(img: Image.Image) -> Image.Image:
    """Crop transparent borders from an RGBA image. Returns original if no bbox found."""
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    alpha = img.split()[-1]
    bbox = alpha.getbbox()
    return img.crop(bbox) if bbox else img


def merge_shoe_pair_image(left_shoe_path: Path, right_shoe_path: Path, output_path: Path, padding: int = 40) -> Path:
    """
    Merge left + right shoe extracted PNGs into a single transparent image (side-by-side).
    This lets us reconstruct/tag a shoe *pair* in one OpenAI call.
    """
    left = Image.open(left_shoe_path).convert("RGBA")
    right = Image.open(right_shoe_path).convert("RGBA")

    left = _crop_to_alpha_bbox(left)
    right = _crop_to_alpha_bbox(right)

    max_h = max(left.height, right.height)
    canvas_w = left.width + right.width + padding
    canvas = Image.new("RGBA", (canvas_w, max_h), (0, 0, 0, 0))

    canvas.paste(left, (0, (max_h - left.height) // 2), left)
    canvas.paste(right, (left.width + padding, (max_h - right.height) // 2), right)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return output_path

# style -> formality
STYLE_TO_FORMALITY = {
    "casual": "casual",
    "sporty": "athletic",
    "athleisure": "athletic",
    "streetwear": "casual",
    "vintage": "casual",
    "minimal": "casual",
    "business": "business-casual",
    "formal": "formal",
    "business-formal": "business-formal",
    "smart-casual": "smart-casual",
    "business-casual": "business-casual",
}

# color keyword -> (family, temperature)
COLOR_MAP = {
    # neutrals
    "black": ("black", "neutral"),
    "white": ("white", "neutral"),
    "gray": ("gray", "neutral"),
    "grey": ("gray", "neutral"),
    "charcoal": ("gray", "neutral"),
    "silver": ("gray", "neutral"),
    "brown": ("brown", "warm"),
    "beige": ("beige", "warm"),
    "tan": ("beige", "warm"),
    "cream": ("beige", "warm"),
    # warm
    "red": ("red", "warm"),
    "maroon": ("maroon", "warm"),
    "burgundy": ("maroon", "warm"),
    "orange": ("orange", "warm"),
    "yellow": ("yellow", "warm"),
    "gold": ("yellow", "warm"),
    "pink": ("pink", "warm"),
    # cool
    "green": ("green", "cool"),
    "olive": ("green", "cool"),
    "teal": ("teal", "cool"),
    "cyan": ("cyan", "cool"),
    "blue": ("blue", "cool"),
    "navy": ("navy", "cool"),
    "purple": ("purple", "cool"),
    "violet": ("purple", "cool"),
    "magenta": ("magenta", "cool"),
}


def _norm(val: str) -> str:
    return (val or "").strip().lower()


def map_garment_role(garment_type: str) -> str:
    return GARMENT_ROLE_MAP.get(_norm(garment_type))


def map_formality(style: str) -> str:
    return STYLE_TO_FORMALITY.get(_norm(style), "casual")


def map_color(primary_color: str):
    pc = _norm(primary_color)
    if pc in COLOR_MAP:
        return COLOR_MAP[pc]
    for key, val in COLOR_MAP.items():
        if key in pc:
            return val
    return ("unknown", "neutral")


def normalize_season(season_value):
    if isinstance(season_value, list):
        seasons = season_value
    elif isinstance(season_value, str):
        seasons = [season_value]
    else:
        seasons = []

    normalized = []
    for s in seasons:
        s_norm = _norm(s)
        if s_norm == "autumn":
            s_norm = "fall"
        if s_norm in SEASON_ENUM and s_norm not in normalized:
            normalized.append(s_norm)
    if not normalized:
        normalized = ["all-season"]
    return normalized


def layering_flags(garment_type: str, sleeve_type: str):
    gt = _norm(garment_type)
    sleeve = _norm(sleeve_type)

    is_base = False
    is_mid = False
    is_outer = False

    if gt in {"jacket", "coat"}:
        is_outer = True
    elif gt in {"t-shirt", "shirt", "blouse", "polo", "tank", "sweater", "hoodie", "dress"}:
        is_base = True
        if "long" in sleeve or "full" in sleeve:
            is_mid = True

    return {
        "is_base_layer": is_base,
        "is_mid_layer": is_mid,
        "is_outer_layer": is_outer,
    }


def enrich_tags(tags: dict) -> dict:
    """
    Add normalized/derived fields without overwriting existing ones.
    Idempotent and deterministic.
    """
    if not isinstance(tags, dict):
        return tags

    # 1) garment_role
    if "garment_role" not in tags:
        role = map_garment_role(tags.get("garment_type"))
        if role:
            tags["garment_role"] = role

    # 2) season array
    if "season_array" not in tags:
        tags["season_array"] = normalize_season(tags.get("season"))

    # 3) color normalization
    fam, temp = map_color(tags.get("primary_color"))
    if "color_family" not in tags:
        tags["color_family"] = fam
    if "color_temperature" not in tags:
        tags["color_temperature"] = temp

    # 4) formality
    if "formality" not in tags:
        tags["formality"] = map_formality(tags.get("style"))

    # 5) layering
    if "layering" not in tags:
        tags["layering"] = layering_flags(tags.get("garment_type"), tags.get("sleeve_type"))

    return tags

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
    image_url = result.data[0].url
    image_response = requests.get(image_url)
    image_response.raise_for_status()
    
    output_path = image_path.parent / f"{image_path.stem}_reconstructed.png"
    with open(output_path, "wb") as f:
        f.write(image_response.content)
    
    logging.info(f"✓ Reconstructed image saved (DALL-E 3): {output_path.name}")
    return output_path


def get_presigned_url(filename: str, bucket_name: str = None, content_type: str = "image/png") -> dict:
    """
    Get presigned URL from Lambda function for S3 upload.
    
    Args:
        filename: Name of the file to upload (object_name in S3)
        bucket_name: Name of the S3 bucket (optional, uses S3_BUCKET_NAME if not provided)
        content_type: MIME type of the file (default: image/png)
    
    Returns:
        dict: Contains 'url' (POST URL) and 'fields' (form fields for POST request)
    """
    if not bucket_name:
        bucket_name = S3_BUCKET_NAME
    
    if not bucket_name:
        raise ValueError("S3 bucket name is required. Set S3_BUCKET_NAME environment variable or pass bucket_name parameter.")
    
    try:
        # Lambda function expects bucket_name and object_name
        # Include Content-Type, Content-Disposition, and ACL in fields and conditions so S3 stores it correctly
        payload = {
            "bucket_name": bucket_name,
            "object_name": filename,
            "expiration": 3600,  # 1 hour expiration
            "fields": {
                "Content-Type": content_type,
                "Content-Disposition": "inline",  # Make browser display instead of download
                "acl": "public-read"  # Make the uploaded object publicly readable
            },
            "conditions": [
                ["starts-with", "$Content-Type", "image/"],
                ["eq", "$Content-Disposition", "inline"],
                ["eq", "$acl", "public-read"]
            ]
        }
        
        response = requests.post(
            LAMBDA_PRESIGNED_URL,
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        
        # Lambda returns wrapped in statusCode/body structure
        lambda_response = response.json()
        
        # Extract the body if it's a string (JSON string)
        if isinstance(lambda_response.get("body"), str):
            presigned_data = json.loads(lambda_response["body"])
        else:
            presigned_data = lambda_response.get("body", lambda_response)
        
        if not presigned_data or "url" not in presigned_data:
            raise ValueError(f"Invalid response from Lambda: {presigned_data}")
        
        return presigned_data
        
    except Exception as e:
        logging.error(f"Failed to get presigned URL: {e}")
        raise RuntimeError(f"Failed to get presigned URL: {e}") from e


def upload_to_s3(file_path: Path, filename: str, bucket_name: str = None) -> str:
    """
    Upload file to S3 using presigned POST URL from Lambda.
    
    Args:
        file_path: Local path to the file
        filename: Name for the file in S3 (object_name)
        bucket_name: Name of the S3 bucket (optional, uses S3_BUCKET_NAME if not provided)
    
    Returns:
        str: Final S3 URL of the uploaded file
    """
    try:
        # Determine Content-Type from file extension
        ext = Path(filename).suffix.lower()
        content_type_map = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.gif': 'image/gif',
            '.webp': 'image/webp',
            '.svg': 'image/svg+xml'
        }
        content_type = content_type_map.get(ext, 'image/png')
        
        # Get presigned POST URL from Lambda with Content-Type
        presigned_data = get_presigned_url(filename, bucket_name, content_type)
        post_url = presigned_data.get("url")
        post_fields = presigned_data.get("fields", {})
        
        if not post_url:
            raise ValueError("No url in Lambda presigned POST response")
        
        # For presigned POST, we need to send:
        # 1. All the fields from presigned_data['fields'] as form data
        # 2. The file itself as the last field (typically 'file')
        # The presigned POST fields usually include: key, AWSAccessKeyId, policy, signature, Content-Type, Content-Disposition, etc.
        
        # Prepare multipart form data
        form_data = post_fields.copy()
        
        # Upload using POST with multipart form data
        # The file should be the last field in the form
        # IMPORTANT: Include Content-Type in the files parameter so S3 stores it correctly
        file_field_name = 'file'  # Default field name for file upload
        
        with open(file_path, "rb") as f:
            # Include Content-Type in the file tuple - this is critical for S3 to store it correctly
            files = {file_field_name: (filename, f, content_type)}
            upload_response = requests.post(
                post_url,
                data=form_data,  # Include all presigned POST fields (including Content-Type and Content-Disposition)
                files=files,     # Include the file with Content-Type (must be last)
                timeout=60
            )
        
        upload_response.raise_for_status()
        
        # Construct the S3 URL
        if not bucket_name:
            bucket_name = S3_BUCKET_NAME
        
        # Construct S3 URL
        s3_region = S3_REGION
        s3_url = f"https://{bucket_name}.s3.{s3_region}.amazonaws.com/{filename}"
        
        logging.info(f"✓ Uploaded to S3: {s3_url}")
        return s3_url
        
    except Exception as e:
        logging.error(f"Failed to upload to S3: {e}")
        raise RuntimeError(f"Failed to upload to S3: {e}") from e


def generate_presigned_get_url(bucket_name: str, object_name: str, expiration: int = 604800) -> str:
    """
    Generate a presigned URL for viewing/downloading an S3 object.
    
    Args:
        bucket_name: Name of the S3 bucket
        object_name: Name of the object in S3
        expiration: Time in seconds for the presigned URL to remain valid (default: 7 days)
    
    Returns:
        str: Presigned URL for accessing the object
    """
    if s3_client is None:
        raise RuntimeError("S3 client not initialized. Check AWS credentials.")
    
    try:
        response = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': bucket_name, 'Key': object_name},
            ExpiresIn=expiration
        )
        return response
    except ClientError as e:
        logging.error(f"Failed to generate presigned URL: {e}")
        raise RuntimeError(f"Failed to generate presigned URL: {e}") from e


def store_tags_in_mongodb(s3_url: str, tags: dict) -> str:
    """
    Store tags in MongoDB with S3 URL.
    
    Args:
        s3_url: S3 URL of the image
        tags: Tags dictionary to store
    
    Returns:
        str: MongoDB document _id
    """
    if mongodb_collection is None:
        raise RuntimeError("MongoDB not connected")
    
    try:
        document = {
            "_id": str(uuid.uuid4()),
            "url": s3_url,
            "tags": tags,
            "created_at": datetime.now(timezone.utc)
        }
        
        result = mongodb_collection.insert_one(document)
        logging.info(f"✓ Stored tags in MongoDB: {result.inserted_id}")
        return str(result.inserted_id)
        
    except Exception as e:
        logging.error(f"Failed to store tags in MongoDB: {e}")
        raise RuntimeError(f"Failed to store tags in MongoDB: {e}") from e


@app.route("/wardrobe-items", methods=["GET"])
def get_wardrobe_items():
    """
    List all wardrobe items stored in MongoDB.
    Returns documents created by store_tags_in_mongodb:
      {
        "_id": string,
        "url": string,
        "tags": dict,
        "created_at": ISO 8601 string
      }
    """
    if mongodb_collection is None:
        return jsonify({"error": "MongoDB not connected"}), 500

    try:
        cursor = mongodb_collection.find().sort("created_at", -1)
        items = []
        for doc in cursor:
            item = {
                "_id": str(doc.get("_id")),
                "url": doc.get("url"),
                "tags": doc.get("tags", {}),
            }
            created_at = doc.get("created_at")
            if isinstance(created_at, datetime):
                item["created_at"] = created_at.isoformat()
            else:
                item["created_at"] = None
            items.append(item)

        return jsonify({"items": items})
    except Exception as e:
        logging.error(f"Failed to fetch wardrobe items from MongoDB: {e}")
        return jsonify({"error": "Failed to fetch wardrobe items"}), 500


@app.route("/recommended-outfits", methods=["GET"])
def get_recommended_outfits():
    """
    List recommended outfits stored in MongoDB.
    Expected document shape in recommended_outfits collection:
      {
        "_id": ObjectId,
        "anchor_item_id": string,
        "outfit_id": int,
        "outfit_score": float,
        "occasion": string,
        "season": string,
        "top": {...},
        "bottom": {...},
        "shoes": {...},
        "outerwear": {...} | null,
        "sunglasses": {...} | null,
        "created_at": datetime
      }
    """
    if outfits_collection is None:
        return jsonify({"error": "MongoDB not connected"}), 500

    try:
        cursor = outfits_collection.find().sort("created_at", -1)
        items = []
        for doc in cursor:
            item = {
                "_id": str(doc.get("_id")),
                "anchor_item_id": doc.get("anchor_item_id"),
                "outfit_id": doc.get("outfit_id"),
                "outfit_score": doc.get("outfit_score"),
                "occasion": doc.get("occasion"),
                "season": doc.get("season"),
                "top": doc.get("top"),
                "bottom": doc.get("bottom"),
                "shoes": doc.get("shoes"),
                "outerwear": doc.get("outerwear"),
                "sunglasses": doc.get("sunglasses"),
            }
            created_at = doc.get("created_at")
            if isinstance(created_at, datetime):
                item["created_at"] = created_at.isoformat()
            else:
                item["created_at"] = None
            items.append(item)

        return jsonify({"items": items})
    except Exception as e:
        logging.error(f"Failed to fetch recommended outfits from MongoDB: {e}")
        return jsonify({"error": "Failed to fetch recommended outfits"}), 500


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
    prompt = """Analyze the clothing item or accessory in this image (this includes ALL types of garments: shirts, pants, shoes, hats, gloves, sunglasses, socks, scarves, dresses, coats, jumpsuits, and any other clothing items or fashion accessories).

Return STRICT JSON only. No markdown, no explanations, no code blocks. Just valid JSON.

The JSON must have exactly this structure:
{
  "garment_type": "string (e.g., T-Shirt, Dress, Pants, Jacket, Shoe, Hat, Glove, Sunglasses, Sock, Scarf, etc.)",
  "sub_category": "string (e.g., T-Shirt -> Crew Neck, V-Neck, Henley; Shoe -> Sneakers, Boots, Loafers; Hat -> Baseball Cap, Beanie, etc.)",
  "primary_color": "string (e.g., navy blue, white, black)",
  "secondary_colors": ["string array of additional colors if any"],
  "fabric": "string (best guess: cotton, polyester, denim, silk, wool, leather, canvas, rubber, etc. - use 'N/A' for items where fabric doesn't apply like shoes)",
  "texture": "string (best guess: smooth, ribbed, textured, knit, etc. - use 'N/A' if not applicable)",
  "pattern": "string (solid, striped, floral, geometric, plaid, etc. or 'none' if solid)",
  "sleeve_type": "string (long sleeve, short sleeve, sleeveless, cap sleeve, etc. - use 'N/A' if not applicable like for shoes, pants, etc.)",
  "fit": "string (slim, regular, loose, oversized, fitted, etc. - use 'N/A' if not applicable)",
  "style": "string (casual, formal, sporty, vintage, modern, athletic, etc.)",
  "season": "string (spring, summer, fall, winter, all-season)",
  "gender": "men" | "women" | "unisex",
  "keywords": ["array of relevant descriptive keywords"]
}

Important:
- Return ONLY the JSON object, nothing else
- All string values must be non-empty (use 'N/A' for fields that don't apply to the item type)
- secondary_colors and keywords can be empty arrays [] if none apply
- Use best-guess inference for fabric and texture if uncertain
- For shoes, accessories, and other non-fabric items, set fabric to 'N/A' and use appropriate material terms
- For items without sleeves (shoes, pants, hats, etc.), set sleeve_type to 'N/A'
- Analyze only the clothing item or accessory, not background or other elements"""

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
        
        # Enrich/normalize without overwriting existing fields
        tags = enrich_tags(tags)

        logging.info(f"✓ Successfully generated tags for {image_path_obj.name}")
        return tags
        
    except FileNotFoundError:
        raise
    except ValueError:
        raise
    except Exception as e:
        logging.error(f"OpenAI API call failed: {e}")
        raise RuntimeError(f"Failed to generate garment tags: {e}") from e


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

            # If both left and right shoes exist, merge them into a single "shoe pair" image
            # so we only do one reconstruction + tagging call for footwear.
            try:
                left_idx = CLOTHING_ITEMS.get("Left-shoe")
                right_idx = CLOTHING_ITEMS.get("Right-shoe")
                left_shoe_path = None
                right_shoe_path = None

                for p in items_to_reconstruct:
                    try:
                        idx = int(p.stem.split("_")[0])
                    except Exception:
                        continue
                    if idx == left_idx:
                        left_shoe_path = p
                    elif idx == right_idx:
                        right_shoe_path = p

                if left_shoe_path and right_shoe_path:
                    merged_name = f"{left_idx:02d}_shoe_pair.png"
                    merged_path = clothes_dir / merged_name
                    merge_shoe_pair_image(left_shoe_path, right_shoe_path, merged_path)

                    items_to_reconstruct = [
                        p for p in items_to_reconstruct if p not in (left_shoe_path, right_shoe_path)
                    ]
                    items_to_reconstruct.append(merged_path)
                    logging.info(
                        f"✓ Merged shoes into one item for reconstruction: {merged_path.name}"
                    )
            except Exception as e:
                logging.warning(f"Failed to merge left/right shoes; continuing separately: {e}")
            
            def _item_name_for_path(item_path: Path) -> str:
                # Extract item name from CLOTHING_ITEMS dictionary
                try:
                    idx_str = item_path.stem.split('_')[0]
                    idx = int(idx_str)
                    item_name = None
                    for name, item_idx in CLOTHING_ITEMS.items():
                        if item_idx == idx:
                            item_name = name
                            break
                    if not item_name:
                        item_name = item_path.stem.split('_', 1)[1] if '_' in item_path.stem else item_path.stem
                except (ValueError, IndexError):
                    item_name = item_path.stem.split('_', 1)[1] if '_' in item_path.stem else item_path.stem

                # Special-case: if we created a merged shoe pair image, label it as a pair
                if "shoe_pair" in item_path.stem.lower():
                    item_name = "Shoes"
                return item_name

            def _process_one_item(item_path: Path) -> dict:
                item_name = _item_name_for_path(item_path)
                logging.info(f"Processing: {item_name}")

                # Stage 2: Clean garment isolation
                cleaned_path = clean_garment_isolation(item_path)

                # Stage 3: Reconstruct with OpenAI
                reconstructed_path = reconstruct_with_openai(cleaned_path, item_name)

                # Stage 4: Tags
                tags = None
                try:
                    logging.info(f"Generating tags for: {item_name}")
                    tags = generate_garment_tags(str(reconstructed_path))
                    logging.info(f"✓ Tags generated for: {item_name}")
                except Exception as tag_error:
                    logging.warning(f"Failed to generate tags for {item_name}: {tag_error}")

                # Stage 5: Upload + Mongo
                s3_url = None
                mongodb_id = None
                try:
                    unique_filename = f"{uuid.uuid4()}_{item_name}_{reconstructed_path.name}"
                    logging.info(f"Uploading {item_name} to S3...")
                    s3_url = upload_to_s3(reconstructed_path, unique_filename)
                    logging.info(f"✓ Uploaded {item_name} to S3: {s3_url}")

                    if tags:
                        mongodb_id = store_tags_in_mongodb(s3_url, tags)
                        logging.info(f"✓ Stored tags in MongoDB: {mongodb_id}")
                except Exception as upload_error:
                    logging.error(f"Failed to upload {item_name} to S3 or store in MongoDB: {upload_error}")

                return {
                    "name": item_name,
                    "path": reconstructed_path,
                    "tags": tags,
                    "s3_url": s3_url,
                    "mongodb_id": mongodb_id,
                }

            # Process each item in parallel (bounded) to reduce total time.
            reconstructed_images = []
            max_workers = int(os.getenv("RECONSTRUCT_MAX_WORKERS", "3"))
            max_workers = max(1, min(max_workers, len(items_to_reconstruct)))

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(_process_one_item, p): p for p in items_to_reconstruct}
                for fut in as_completed(futures):
                    item_path = futures[fut]
                    try:
                        info = fut.result()
                        reconstructed_images.append(info)
                        logging.info(f"✓ Reconstructed: {info.get('name')}")
                    except Exception as e:
                        logging.warning(f"Failed to reconstruct {item_path.name}: {e}")
                        continue
            
            if not reconstructed_images:
                return jsonify({"error": "Failed to reconstruct any clothing items"}), 500
            
            # Build response with S3 URLs (only for reconstructed images)
            reconstructed_data = []
            for img_info in reconstructed_images:
                    item_data = {
                        "name": img_info["name"],
                    "s3_url": img_info.get("s3_url"),
                    "mongodb_id": img_info.get("mongodb_id")
                    }
                    # Include tags if available
                    if img_info.get("tags"):
                        item_data["tags"] = img_info["tags"]
                    reconstructed_data.append(item_data)
            
            logging.info(f"✓ Pipeline complete - processed {len(reconstructed_images)} reconstructed images")
            
            return jsonify({
                "success": True,
                "message": f"Successfully processed {len(reconstructed_images)} clothing items",
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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
