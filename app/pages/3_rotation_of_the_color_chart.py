import streamlit as st
import cv2
import numpy as np
from streamlit_extras.switch_page_button import switch_page


def rotate_image(image, degrees, padding):
    """Rotates an image using OpenCV.

    Args:
        image: The cv2 object to rotate.
        degrees: The rotation angle in degrees.
        padding: The padding to add to the image.

    Returns:
        The rotated image as a cv2 object.
    """

    # Convert the PIL Image to a NumPy array for OpenCV
    img_array = np.array(image)

    # add padding to the image to avoid cropping
    img_array = cv2.copyMakeBorder(img_array, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=[255, 255, 255])

    # Rotate the image using OpenCV's rotation matrix
    height, width = img_array.shape[:2]
    center = (width // 2, height // 2)
    matrix = cv2.getRotationMatrix2D(center, degrees, 1.0)
    rotated_img_array = cv2.warpAffine(img_array, matrix, (width, height))

    # Convert the rotated NumPy array back to a PIL Image
    # rotated_image = Image.fromarray(rotated_img_array)

    return rotated_img_array

def switch_to_color():
    """Must be one of ['streamlit starting page', 'upload image and define areas', 'build custom color chart', 'color analysis and mapping', 'rotation of the color chart']"""

    want_to_contribute = st.button("Go back to Separate the color chart segments?")
    if want_to_contribute:
        switch_page("build custom color chart")

def switch_to_cropping():
    """Must be one of ['streamlit starting page', 'upload image and define areas', 'build custom color chart', 'color analysis and mapping', 'rotation of the color chart']"""

    want_to_contribute = st.button("Upload the image?")
    if want_to_contribute:
        switch_page("upload image and define areas")

def switch_to_ruler():
    """Navigate back to ruler selection page"""
    want_to_contribute = st.button("← Back to Ruler Selection")
    if want_to_contribute:
        switch_page("select ruler")

def is_color_chart_in_session_state():
    if "chart_img" not in st.session_state:

        st.write("Please go to page 1 to upload the image")
        switch_to_cropping()
    else:
        st.write("Color chart image is already in session state")

# ... (code to display the rotated image)
def main():
    st.title("Rotation Tool")
    
    # Mode selection - determine what we're working with
    working_mode = None
    if "ruler_img" in st.session_state and "chart_img" in st.session_state:
        # Both available - let user choose
        st.info("🔀 Both Chart and Ruler images are available. Select which one to rotate:")
        working_mode = st.radio("Working with:", ["Color Chart", "Ruler"], horizontal=True)
    elif "ruler_img" in st.session_state:
        working_mode = "Ruler"
        st.info("📏 Working with: **Ruler Image**")
    elif "chart_img" in st.session_state:
        working_mode = "Color Chart"
        st.info("🎨 Working with: **Color Chart**")
    else:
        st.warning("⚠️ No image found in memory.")
        col1, col2 = st.columns(2)
        with col1:
            st.write("Upload a color chart:")
            switch_to_cropping()
        with col2:
            st.write("Upload a ruler:")
            switch_to_ruler()
        return
    
    # Get the appropriate image based on mode
    if working_mode == "Ruler":
        image_key = "ruler_img"
        title_text = "Ruler"
        save_button_text = "💾 Save Ruler to Memory"
    else:
        image_key = "chart_img"
        title_text = "Color Chart"
        save_button_text = "💾 Save Chart to Memory"
    
    image = st.session_state[image_key]
    
    st.markdown(f"### Rotating the {title_text}")
    st.markdown("You can rotate the image to the desired angle and add padding to avoid cropping.")
    st.markdown("After rotating the image, save it back to memory.")
    
    # Rotation controls
    with st.sidebar:
        st.header("Rotation Controls")
        rotation_angle = st.slider("Rotate image:", min_value=-180, max_value=180, value=0)
        padding = st.slider("Padding", min_value=0, max_value=500, value=100)
    
    # Apply rotation
    rotated_image = rotate_image(image, rotation_angle, padding)
    st.image(rotated_image, caption=f"{title_text} (Rotated)", use_column_width=True)
    
    # Save button
    col1, col2 = st.columns([2, 1])
    with col1:
        if st.button(save_button_text, type="primary", use_container_width=True):
            st.session_state[image_key] = rotated_image
            st.success(f"✅ {title_text} saved to memory!")
            st.balloons()
    
    # Navigation
    st.markdown("---")
    st.header("Navigation")
    if working_mode == "Ruler":
        switch_to_ruler()
    else:
        switch_to_color()


# Streamlit app execution
if __name__ == '__main__':
    main()
