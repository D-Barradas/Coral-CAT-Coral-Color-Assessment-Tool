import streamlit as st
import cv2
import numpy as np
from streamlit_extras.switch_page_button import switch_page


def rotate_image(image, degrees,padding):
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
    """Must be one of ['streamlit starting page', 'cropping color chart', 'separate color chart', 'analysis and mapping', 'rotation of the color chart']"""

    want_to_contribute = st.button("Go back to Separate the color chart segments?")
    if want_to_contribute:
        switch_page("separate color chart")

def switch_to_cropping():
    """Must be one of ['streamlit starting page', 'cropping color chart', 'separate color chart', 'analysis and mapping', 'rotation of the color chart']"""

    want_to_contribute = st.button("Upload the image?")
    if want_to_contribute:
        switch_page("cropping color chart")

def is_color_chart_in_session_state():
    if "chart_img" not in st.session_state:

        st.write("Please go to page 1 to upload the image")
        switch_to_cropping()
    else:
        st.write("Color chart image is already in session state")

# ... (code to display the rotated image)
def main():
    st.title("Rotation of the color chart")
    if "chart_img" in st.session_state:
        st.title("Rotation of the color chart")
        st.markdown("In this page, you can rotate the color chart to the desired angle.")
        st.markdown("You can also add padding to the image to avoid cropping.")
        st.markdown("After rotating the image, you can save the image to memory.")
        st.markdown("You can also go back to the previous page to separate the color chart segments.")

        is_color_chart_in_session_state()
        # image = st.session_state["chart_img"]
        # ... (code to create a slider for rotation)
        with st.sidebar:

            rotation_angle = st.slider("Rotate image:", min_value=-180, max_value=180)
            padding = st.slider("Padding", min_value=0, max_value=500, value=100)
        # ... (code to apply the rotation)
        rotated_image = rotate_image(st.session_state["chart_img"], rotation_angle ,padding)
        st.image(rotated_image, caption="Color Chart to Rotate", use_column_width=True)
        if st.button("Save Images to memory"):
            st.session_state["chart_img"] = rotated_image
        switch_to_color()
    else:
        st.write("Please go to page 1 to upload the image and select the color chart")
        switch_to_cropping()

# Streamlit app execution
if __name__ == '__main__':
    main()
