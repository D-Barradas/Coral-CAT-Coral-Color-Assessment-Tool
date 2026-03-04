import streamlit as st
import cv2
import numpy as np
import pandas as pd
from streamlit_extras.image_selector import image_selector, show_selection
from streamlit_extras.switch_page_button import switch_page

st.set_page_config(page_title="Select Ruler", page_icon="📏")


import io
import zipfile
from load_functions import crop_my_image
def switch_to_next():
    """Must be one of ['streamlit starting page', 'upload image and define areas', 'build custom color chart', 'color analysis and mapping', 'rotation of the color chart']"""

    want_to_contribute = st.button("Go next phase?")
    if want_to_contribute:
        switch_page("color analysis and mapping")

def switch_to_cropping():
    """Must be one of ['streamlit starting page', 'upload image and define areas', 'build custom color chart', 'color analysis and mapping', 'rotation of the color chart']"""

    want_to_contribute = st.button("Upload the image?")
    if want_to_contribute:
        switch_page("upload image and define areas")

def _get_ruler_records():
    if "ruler_records" not in st.session_state:
        st.session_state["ruler_records"] = []
    return st.session_state["ruler_records"]


def main():
    st.title("Select ruler and record measurement")

    if "chart_img" not in st.session_state and "coral_img" not in st.session_state:
        st.write("Please upload an image and select chart/coral areas on page 1 first.")
        switch_to_cropping()


    else :
        # prefer rotated/processed chart image if available
        img = st.session_state.get("chart_img") # or st.session_state.get("coral_img")

        st.markdown("Select a rectangular box around the physical ruler in the image.")
        selection = image_selector(image=img, selection_type="box", key="ruler_box")

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

                st.write(f"Selected box: x={x0},{x1} y={y0},{y1} — px_width={px_w} px_height={px_h}")

                # preview crop
                crop = img[y0:y1, x0:x1].copy()
                st.image(crop, caption="Ruler crop preview", use_column_width=False)

                # measurement choices
                mm_choice = st.radio("Physical length to assign to this selection:",
                                    ("Full bar (8 mm)", "Color portion (6.4 mm)", "Square (0.9 mm)", "Custom mm"))
                if mm_choice == "Full bar (8 mm)":
                    mm_value = 8.0
                elif mm_choice == "Color portion (6.4 mm)":
                    mm_value = 6.4
                elif mm_choice == "Square (0.9 mm)":
                    mm_value = 0.9
                else:
                    mm_value = st.number_input("Enter physical length in mm:", min_value=0.0, value=8.0)

                original_filename = st.text_input("Original image filename (for linking to SAM outputs, e.g. PB300143.JPG)")

                if st.button("Save ruler"):
                    mm_per_px = float(mm_value) / float(max(1, px_length))

                    crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
                    ok, encoded = cv2.imencode(".png", crop_bgr)
                    crop_bytes = encoded.tobytes() if ok else None

                    rec = {
                        "image": original_filename,
                        "px_length": int(px_length),
                        "mm_value": float(mm_value),
                        "mm_per_px": float(mm_per_px),
                        "crop_bytes": crop_bytes,
                    }

                    records = _get_ruler_records()
                    records.append(rec)

                    # also store latest in session state
                    st.session_state["ruler"] = rec

                    st.success(f"Saved ruler — mm_per_px={mm_per_px:.6f}")

    st.markdown("---")
    st.header("Downloads")

    records = _get_ruler_records()
    if records:
        df = pd.DataFrame([{k: v for k, v in r.items() if k != "crop_bytes"} for r in records])
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download rulers CSV", data=csv_bytes, file_name="rulers.csv", mime="text/csv")

        # allow choosing a record to download its crop + CSV in a zip
        images = df["image"].fillna("").astype(str).tolist()
        if len(images) > 0:
            sel = st.selectbox("Select record to package", options=list(dict.fromkeys(images)))
            if sel:
                rec = next((r for r in reversed(records) if r.get("image") == sel), None)
                crop_bytes = rec.get("crop_bytes") if rec else None
                if crop_bytes:
                    bio = io.BytesIO()
                    with zipfile.ZipFile(bio, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
                        zf.writestr("rulers.csv", csv_bytes)
                        safe_name = f"{sel}_ruler_crop.png" if sel else "ruler_crop.png"
                        zf.writestr(safe_name, crop_bytes)
                    bio.seek(0)
                    st.download_button(
                        f"Download zip for {sel}",
                        data=bio.getvalue(),
                        file_name=f"{sel}_ruler.zip",
                        mime="application/zip",
                    )
                else:
                    st.write("No crop bytes available for selected record.")
    else:
        st.write("No ruler records in session yet. Save a ruler first to enable downloads.")



if __name__ == '__main__':
    main()
