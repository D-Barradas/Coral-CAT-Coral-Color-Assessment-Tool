import streamlit as st

# MUST be first Streamlit command
st.set_page_config(page_title="Upload & Select Ruler", page_icon="📏")

# Now import everything else
import cv2
import numpy as np
import pandas as pd
import io
import zipfile
from pathlib import Path
from datetime import datetime

from streamlit_extras.image_selector import image_selector, show_selection
from streamlit_extras.switch_page_button import switch_page
from load_functions import crop_my_image

def switch_to_rotation():
    """Navigate to rotation page for ruler modification"""
    want_to_contribute = st.button("Rotate Ruler Image →")
    if want_to_contribute:
        switch_page("rotation of the color chart")

def switch_to_dewarp():
    """Navigate to dewarp page for ruler modification"""
    want_to_contribute = st.button("Dewarp Ruler Image →")
    if want_to_contribute:
        switch_page("dewarp the image")

def _get_ruler_records():
    if "ruler_records" not in st.session_state:
        st.session_state["ruler_records"] = []
    return st.session_state["ruler_records"]

def _save_ruler_csv(records):
    """Save ruler records to CSV file"""
    csv_path = Path("../data/interim/ruler_detection_results.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "crop_bytes"} for r in records])
    df.to_csv(csv_path, index=False)
    return csv_path


def main():
    st.title("Upload Images & Select Ruler")
    
    st.markdown("""
    ### Workflow:
    1. Upload one or more images containing rulers
    2. Select an image to work with
    3. Draw bounding boxes around rulers (can select multiple from same image)
    4. Optionally modify full image (rotate/dewarp) before selecting rulers
    5. Save each ruler measurement individually
    """)
    
    # Step 1: Image upload
    st.markdown("---")
    st.header("Step 1: Upload Images")
    uploaded_files = st.file_uploader(
        "Upload images with rulers",
        type=["jpg", "jpeg", "png", "bmp", "tiff"],
        accept_multiple_files=True,
        key="ruler_uploader"
    )
    
    if not uploaded_files:
        st.info("📤 Please upload at least one image to begin")
        return
    
    # Store uploaded images in session state
    if "uploaded_ruler_images" not in st.session_state:
        st.session_state["uploaded_ruler_images"] = {}
    
    # Store modified versions separately
    if "modified_ruler_images" not in st.session_state:
        st.session_state["modified_ruler_images"] = {}
    
    # Process uploaded files
    for uploaded_file in uploaded_files:
        if uploaded_file.name not in st.session_state["uploaded_ruler_images"]:
            # Read image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            st.session_state["uploaded_ruler_images"][uploaded_file.name] = img_rgb
    
    # Step 2: Image selection
    st.markdown("---")
    st.header("Step 2: Select Image to Work With")
    
    image_names = list(st.session_state["uploaded_ruler_images"].keys())
    
    # Initialize selected image if not set
    if "selected_ruler_image_name" not in st.session_state or st.session_state["selected_ruler_image_name"] not in image_names:
        st.session_state["selected_ruler_image_name"] = image_names[0]
    
    selected_image_name = st.session_state["selected_ruler_image_name"]
    
    # Display thumbnails for selection
    cols = st.columns(min(4, len(image_names)))
    
    for idx, img_name in enumerate(image_names):
        with cols[idx % 4]:
            img_thumb = st.session_state["uploaded_ruler_images"][img_name]
            st.image(img_thumb, caption=img_name, use_column_width=True)
            if st.button(f"Select", key=f"select_{img_name}"):
                st.session_state["selected_ruler_image_name"] = img_name
                selected_image_name = img_name
                st.rerun()
    
    st.info(f"🖼️ Currently working with: **{selected_image_name}**")
    
    # Check if there's a modified version from rotation/dewarp
    # The modified version is returned from pages 3/6 via ruler_img
    if "ruler_img" in st.session_state and st.session_state.get("ruler_source_image") == selected_image_name:
        # Update the modified version
        st.session_state["modified_ruler_images"][selected_image_name] = st.session_state["ruler_img"]
    
    # Determine which image to use: modified or original
    has_modified = selected_image_name in st.session_state["modified_ruler_images"]
    use_modified = True  # Default value
    
    if has_modified:
        use_modified = st.checkbox(
            "📐 Use modified version (rotated/dewarped)", 
            value=True, 
            help="Uncheck to work with the original uploaded image"
        )
        if use_modified:
            img = st.session_state["modified_ruler_images"][selected_image_name]
            st.success("✨ Working with modified image")
        else:
            img = st.session_state["uploaded_ruler_images"][selected_image_name]
            st.info("📷 Working with original image")
    else:
        img = st.session_state["uploaded_ruler_images"][selected_image_name]
        st.info("📷 Working with original image (no modifications yet)")
    
    # Image modification options (before ruler selection)
    with st.expander("🔧 Modify Full Image (Optional)", expanded=False):
        st.markdown("Apply transformations to the entire image before selecting rulers:")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Rotate Full Image", use_container_width=True):
                # Store full image in ruler_img for modification
                st.session_state["ruler_img"] = img.copy()
                st.session_state["ruler_source_image"] = selected_image_name
                switch_page("rotation of the color chart")
        
        with col2:
            if st.button("📐 Dewarp Full Image", use_container_width=True):
                # Store full image in ruler_img for modification
                st.session_state["ruler_img"] = img.copy()
                st.session_state["ruler_source_image"] = selected_image_name
                switch_page("dewarp the image")
        
        with col3:
            if has_modified and st.button("🔙 Reset to Original", use_container_width=True):
                del st.session_state["modified_ruler_images"][selected_image_name]
                if "ruler_img" in st.session_state:
                    del st.session_state["ruler_img"]
                st.rerun()
    
    # Display count of rulers selected from this image
    records = _get_ruler_records()
    rulers_from_this_image = [r for r in records if r.get("image") == selected_image_name]
    ruler_count = len(rulers_from_this_image)
    
    if rulers_from_this_image:
        st.success(f"✅ {ruler_count} ruler(s) already selected from this image")
    
    # Step 3: Ruler selection
    st.markdown("---")
    st.header("Step 3: Select Ruler Area")
    st.markdown("Draw a rectangular box around a ruler in the image. You can select multiple rulers from the same image.")
    
    # Use a stable key for the selector
    selector_key = f"ruler_box_{selected_image_name}"
    selection = image_selector(image=img, selection_type="box", key=selector_key)

    if selection:
        try:
            show_selection(img, selection)
        except Exception:
            pass

        if len(selection.get("selection", {}).get("box", [])):
            bx = selection["selection"]["box"][0]
            x0, x1 = int(bx["x"][0]), int(bx["x"][1])
            y0, y1 = int(bx["y"][0]), int(bx["y"][1])
            px_w = abs(x1 - x0)
            px_h = abs(y1 - y0)
            px_length = max(px_w, px_h)

            st.write(f"**Selection Details:** x=[{x0}, {x1}] y=[{y0}, {y1}] | Width={px_w}px Height={px_h}px")

            # Preview crop
            crop = img[y0:y1, x0:x1].copy()
            st.image(crop, caption="Ruler Crop Preview", use_column_width=False)
            
            # Step 4: Measurement specification
            st.markdown("---")
            st.header("Step 4: Specify Physical Measurement")

            # Measurement choices - use stable key
            radio_key = f"mm_choice_{selected_image_name}"
            mm_choice = st.radio("Physical length to assign to this selection:",
                                ("Full bar (8 mm)", "Color portion (6.4 mm)", "Square (0.9 mm)", "Custom mm"),
                                key=radio_key)
            if mm_choice == "Full bar (8 mm)":
                mm_value = 8.0
            elif mm_choice == "Color portion (6.4 mm)":
                mm_value = 6.4
            elif mm_choice == "Square (0.9 mm)":
                mm_value = 0.9
            else:
                mm_value = st.number_input("Enter physical length in mm:", min_value=0.0, value=8.0)

            # Step 5: Save and calculate
            st.markdown("---")
            st.header("Step 5: Save Ruler Measurement")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("💾 Save This Ruler", type="primary", use_container_width=True):
                    # Automatic calculation
                    mm_per_px = float(mm_value) / float(max(1, px_length))
                    
                    # Store cropped ruler in a temporary variable for modification
                    st.session_state["ruler_img"] = crop.copy()
                    st.session_state["ruler_source_image"] = selected_image_name
                    st.session_state["ruler_bbox"] = {"x0": x0, "x1": x1, "y0": y0, "y1": y1}

                    crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
                    ok, encoded = cv2.imencode(".png", crop_bgr)
                    crop_bytes = encoded.tobytes() if ok else None

                    rec = {
                        "image": selected_image_name,
                        "ruler_number": ruler_count + 1,
                        "was_modified": has_modified and use_modified if has_modified else False,
                        "px_width": int(px_w),
                        "px_height": int(px_h),
                        "px_length": int(px_length),
                        "mm_value": float(mm_value),
                        "mm_per_px": float(mm_per_px),
                        "bbox_x0": x0,
                        "bbox_y0": y0,
                        "bbox_x1": x1,
                        "bbox_y1": y1,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "crop_bytes": crop_bytes,
                    }

                    records = _get_ruler_records()
                    records.append(rec)

                    # Save to CSV
                    csv_path = _save_ruler_csv(records)

                    # Also store latest in session state
                    st.session_state["ruler"] = rec

                    st.success(f"✅ Ruler #{ruler_count + 1} saved! Conversion: **{mm_per_px:.6f} mm/px**")
                    st.info(f"📁 Data saved to: `{csv_path}`")
                    st.info("🔄 You can now select another ruler from this image or switch images")
                    
                    # Rerun to update the counter
                    st.rerun()
            
            with col2:
                st.metric("Conversion Factor", f"{mm_value / max(1, px_length):.6f} mm/px")
                st.metric("Ruler Area", f"{px_w * px_h:,} px²")
    
    else:
        st.warning("👆 Please draw a bounding box around a ruler in the image above")
    
    # Downloads section
    st.markdown("---")
    st.header("📥 Downloads & Summary")

    records = _get_ruler_records()
    if records:
        # Show summary statistics
        df = pd.DataFrame([{k: v for k, v in r.items() if k != "crop_bytes"} for r in records])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rulers", len(df))
        with col2:
            st.metric("Images Processed", df['image'].nunique())
        with col3:
            avg_conversion = df['mm_per_px'].mean()
            st.metric("Avg Conversion", f"{avg_conversion:.6f} mm/px")
        
        # Show detailed table
        st.subheader("All Ruler Measurements")
        st.dataframe(df, use_container_width=True)
        
        # Download options
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.download_button(
                "📄 Download All Rulers CSV", 
                data=csv_bytes, 
                file_name=f"rulers_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", 
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            # Allow choosing a record to download its crop + CSV in a zip
            images = df["image"].fillna("").astype(str).tolist()
            if len(images) > 0:
                sel = st.selectbox("Select image to download all its rulers:", options=list(dict.fromkeys(images)))
                if sel:
                    # Get all records for this image
                    image_records = [r for r in records if r.get("image") == sel]
                    if image_records:
                        bio = io.BytesIO()
                        with zipfile.ZipFile(bio, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
                            # Add CSV with just this image's rulers
                            image_df = df[df['image'] == sel]
                            zf.writestr(f"{sel}_rulers.csv", image_df.to_csv(index=False))
                            
                            # Add all ruler crops for this image
                            for idx, rec in enumerate(image_records):
                                crop_bytes = rec.get("crop_bytes")
                                if crop_bytes:
                                    ruler_num = rec.get("ruler_number", idx + 1)
                                    safe_name = f"{sel}_ruler_{ruler_num}.png"
                                    zf.writestr(safe_name, crop_bytes)
                        
                        bio.seek(0)
                        st.download_button(
                            f"📦 Download ZIP for {sel} ({len(image_records)} ruler(s))",
                            data=bio.getvalue(),
                            file_name=f"{sel}_rulers.zip",
                            mime="application/zip",
                            use_container_width=True
                        )
    else:
        st.info("No ruler records in session yet. Save a ruler first to enable downloads.")


if __name__ == '__main__':
    main()
