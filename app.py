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
import logging
import os
import subprocess
import tempfile
import zipfile
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv
from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from PIL import Image
from openai import OpenAI

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


def reconstruct_with_openai(image_path: Path) -> Path:
    """
    Stage 3: Reconstruct clothing using OpenAI Image API.
    Uses gpt-image-1 model for image-to-image reconstruction.
    """
    logging.info("Stage 3: Reconstructing with OpenAI API...")
    
    # Prompt for reconstruction
    prompt = """
    Reconstruct this garment as a high-quality ecommerce product image.
    - Lay it completely flat
    - Symmetric shape
    - Remove all wrinkles and folds
    - Preserve fabric texture exactly
    - Preserve all colors, patterns, logos, and text exactly
    - Centered composition
    - Pure white background (RGB 255, 255, 255)
    - No mannequin, no human body, no shadows
    - Professional product photography style
    """
    
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


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "message": "Clothing reconstruction API is running"})


@app.route("/reconstruct", methods=["POST"])
def reconstruct():
    """
    Main endpoint: Full pipeline from image upload to reconstructed product images.
    
    Extracts and reconstructs all clothing items found in the image:
    - Hat, Glove, Sunglasses, Upper-clothes, Dress, Coat
    - Socks, Pants, Jumpsuits, Scarf, Left-shoe, Right-shoe
    
    Request:
        - Form data with 'file' field containing image
    
    Response:
        - JSON with base64 encoded images:
          {
            "originalImage": "data:image/...;base64,...",
            "maskImage": "data:image/png;base64,...",
            "reconstructedImages": [
              {"name": "item_name", "image": "data:image/png;base64,..."},
              ...
            ]
          }
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
            
            # Process each item: clean and reconstruct
            reconstructed_images = []
            for item_path in items_to_reconstruct:
                item_name = item_path.stem.split('_', 1)[1] if '_' in item_path.stem else item_path.stem
                logging.info(f"Processing: {item_name}")
                
                try:
                    # Stage 2: Clean garment isolation
                    cleaned_path = clean_garment_isolation(item_path)
                    
                    # Stage 3: Reconstruct with OpenAI
                    reconstructed_path = reconstruct_with_openai(cleaned_path)
                    
                    reconstructed_images.append({
                        "name": item_name,
                        "path": reconstructed_path
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
            
            # Reconstructed images
            reconstructed_data = []
            for img_info in reconstructed_images:
                with open(img_info["path"], "rb") as f:
                    img_b64 = base64.b64encode(f.read()).decode('utf-8')
                    reconstructed_data.append({
                        "name": img_info["name"],
                        "image": f"data:image/png;base64,{img_b64}"
                    })
            
            logging.info(f"✓ Pipeline complete - returning {len(reconstructed_images)} reconstructed images")
            
            return jsonify({
                "originalImage": f"data:{mime_type};base64,{original_image_b64}",
                "maskImage": f"data:image/png;base64,{mask_image_b64}",
                "reconstructedImages": reconstructed_data
            })
    
    except Exception as e:
        logging.error(f"API Error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
