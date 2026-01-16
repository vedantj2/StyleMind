# API cURL Examples

## 1. Reconstruct Endpoint (with automatic tagging)

This endpoint processes an image, extracts clothing items, reconstructs them, and automatically generates tags for each reconstructed garment.

```bash
curl -X POST http://localhost:5000/reconstruct \
  -F "file=@path/to/your/image.jpg" \
  -H "Content-Type: multipart/form-data"
```

**Example with a specific image:**
```bash
curl -X POST http://localhost:5000/reconstruct \
  -F "file=@test_image.jpg"
```

**Response includes:**
- `originalImage`: Base64 encoded original image
- `maskImage`: Base64 encoded segmentation mask
- `reconstructedImages`: Array of reconstructed garments, each with:
  - `name`: Item name (e.g., "Upper-clothes", "Pants")
  - `image`: Base64 encoded reconstructed image
  - `tags`: Structured JSON tags (if tagging succeeded)

**Example response:**
```json
{
  "originalImage": "data:image/jpeg;base64,...",
  "maskImage": "data:image/png;base64,...",
  "reconstructedImages": [
    {
      "name": "Upper-clothes",
      "image": "data:image/png;base64,...",
      "tags": {
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
    }
  ]
}
```

---

## 2. Tag Garment Endpoint (standalone)

This endpoint generates tags for an already reconstructed garment image.

### Option A: Upload image file

```bash
curl -X POST http://localhost:5000/tag-garment \
  -F "file=@reconstructed_tshirt.png" \
  -H "Content-Type: multipart/form-data"
```

### Option B: Provide image path (if image is on server)

```bash
curl -X POST http://localhost:5000/tag-garment \
  -H "Content-Type: application/json" \
  -d '{"image_path": "output/reconstructed_tshirt_reconstructed.png"}'
```

**Response:**
```json
{
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
```

---

## 3. Health Check Endpoint

```bash
curl -X GET http://localhost:5000/health
```

**Response:**
```json
{
  "status": "healthy",
  "message": "Clothing reconstruction API is running"
}
```

---

## Notes

- **Server URL**: Default is `http://localhost:5000`. Change if your server runs on a different host/port.
- **API Key**: Make sure `OPENAI_API_KEY` is set in your environment or `.env` file.
- **Image Formats**: Supported formats include JPG, JPEG, PNG, GIF, WebP.
- **File Size**: Be mindful of image file sizes for uploads.

---

## Save Response to File

To save the JSON response to a file:

```bash
curl -X POST http://localhost:5000/reconstruct \
  -F "file=@test_image.jpg" \
  -o response.json
```

Or pretty-print with `jq`:

```bash
curl -X POST http://localhost:5000/reconstruct \
  -F "file=@test_image.jpg" | jq '.'
```

---

## Error Handling

If an error occurs, the API returns a JSON error response:

```json
{
  "error": "Error message here"
}
```

Common HTTP status codes:
- `200`: Success
- `400`: Bad Request (missing file, invalid input)
- `404`: Not Found (image path doesn't exist)
- `500`: Internal Server Error (API key missing, processing failed)

