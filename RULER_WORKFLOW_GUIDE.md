# Ruler Measurement Workflow Guide

## Overview
The enhanced Page 10 now supports a complete workflow for measuring rulers from multiple images, with the ability to pre-process images (rotate/dewarp) and select multiple rulers from each image.

## Features

### ✨ Key Enhancements

1. **Multiple Image Upload** - Upload several images at once
2. **Image Preprocessing** - Rotate or dewarp full images before ruler selection
3. **Multiple Rulers per Image** - Select many rulers from the same image
4. **Modified Image Persistence** - Modifications are preserved when switching between pages
5. **Enhanced CSV Output** - Tracks ruler numbers and modification status

## Workflow Steps

### 1. Upload Images
- Navigate to **Page 10: Upload & Select Ruler**
- Upload one or more images containing rulers
- All images are stored in memory

### 2. Select Working Image
- Thumbnails display all uploaded images
- Click "Select" on the image you want to work with
- The selected image becomes active

### 3. Optional: Preprocess Full Image
- Expand "🔧 Modify Full Image (Optional)"
- Choose to **Rotate** or **Dewarp** the entire image
- This redirects to Page 3 or Page 6
- Make adjustments and click "Save"
- Click "← Back to Ruler Selection" to return to Page 10
- The modified image is now available!

### 4. Work with Original or Modified
- If you modified the image, a checkbox appears:
  - ✅ "Use modified version" (default)
  - Or uncheck to work with original
- Toggle anytime to compare or work with either version

### 5. Select Rulers
- Draw bounding boxes around rulers in the image
- You can select **multiple rulers** from the same image
- Each selection shows:
  - Preview of the cropped ruler
  - Pixel dimensions (width, height)
  
### 6. Specify Measurement
- Choose physical measurement:
  - Full bar (8 mm)
  - Color portion (6.4 mm)
  - Square (0.9 mm)
  - Custom value
  
### 7. Save Each Ruler
- Click "💾 Save This Ruler"
- Conversion factor calculated automatically (mm/px)
- Ruler is saved to CSV with metadata
- Counter updates showing how many rulers from this image
- **You can immediately select another ruler from the same image!**

### 8. Switch Images & Repeat
- Select a different image from the thumbnails
- Repeat the process (optionally modify, select rulers, save)
- All rulers from all images are tracked

### 9. Download Results
- **Summary Statistics**:
  - Total rulers measured
  - Number of images processed
  - Average conversion factor
  
- **CSV Export**: All rulers with full metadata
- **ZIP Export**: Select an image to download:
  - CSV with all rulers from that image
  - PNG crops of each ruler from that image

## Session State Variables

The implementation uses these key session state variables:

- `uploaded_ruler_images` - Original uploaded images (dict)
- `modified_ruler_images` - Modified versions after rotation/dewarp (dict)
- `selected_ruler_image_name` - Currently selected image name (str)
- `ruler_img` - Current working image (for modifications) (numpy array)
- `ruler_source_image` - Tracks which image ruler_img came from (str)
- `ruler_records` - List of all saved ruler measurements (list)

## CSV Output Format

Each saved ruler record includes:

| Column | Description |
|--------|-------------|
| `image` | Original image filename |
| `ruler_number` | Sequential number for this image (1, 2, 3...) |
| `was_modified` | Boolean: was image rotated/dewarped? |
| `px_width` | Ruler width in pixels |
| `px_height` | Ruler height in pixels |
| `px_length` | Maximum dimension (used for conversion) |
| `mm_value` | Physical measurement assigned |
| `mm_per_px` | Calculated conversion factor |
| `bbox_x0`, `bbox_y0`, `bbox_x1`, `bbox_y1` | Bounding box coordinates |
| `timestamp` | When the ruler was saved |

## Tips & Best Practices

1. **Preprocessing First**: If your image needs rotation or dewarping, do it BEFORE selecting rulers
2. **Multiple Rulers**: Don't switch images until you've selected all rulers from the current one
3. **Reset if Needed**: Use "🔙 Reset to Original" to discard modifications
4. **Check the Counter**: The "✅ X ruler(s) already selected" message helps track progress
5. **Download Per Image**: Use ZIP downloads to get all rulers from a specific image organized together

## Navigation Flow

```
Page 10 (Upload) 
    ↓
Select Image
    ↓
[Optional] → Page 3 (Rotate) OR Page 6 (Dewarp) → Save → Back to Page 10
    ↓
Select Ruler #1 → Save
    ↓
Select Ruler #2 → Save
    ↓
Select Ruler #N → Save
    ↓
Switch to Next Image → Repeat
    ↓
Download All Results
```

## Technical Notes

- Images are stored as NumPy arrays in RGB format
- Cropped rulers are encoded as PNG bytes for CSV storage
- CSV updates on each save (incremental)
- No data is lost when switching between pages
- Modified images persist for the entire session
- Unique keys per image prevent selector conflicts

---

**Last Updated**: March 4, 2026
**Version**: 2.0
