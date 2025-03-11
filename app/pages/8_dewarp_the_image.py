import streamlit as st
import cv2
import numpy as np
from streamlit_extras.switch_page_button import switch_page


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

def perspective_correction(image, pts):
    """
    Perform a perspective correction on the given image using the provided points.

    Parameters:
    - image: The input image on which perspective correction is to be applied.
    - pts: A list of four points (top-left, top-right, bottom-right, bottom-left) that define the perspective transform.

    Returns:
    - The perspective-corrected image.
    """
    # Obtain the width and height of the new image
    (tl, tr, br, bl) = pts
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

            # Define the destination points for the perspective transform
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # Compute the perspective transform matrix and apply it
    M = cv2.getPerspectiveTransform(np.array(pts, dtype="float32"), dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped

def select_perspective_points(image_shape):
    """
    Select the four points needed for perspective correction using sliders.

    Parameters:
    - image_shape: The shape of the image (height, width).

    Returns:
    - A list of four points (top-left, top-right, bottom-right, bottom-left).
    """
    height, width = image_shape[:2]

    st.markdown("### Select the points for perspective correction")

    with st.sidebar:

        tl_x = st.slider("Top-Left X", 0, width, 0)
        tl_y = st.slider("Top-Left Y", 0, height, 0)
        tr_x = st.slider("Top-Right X", 0, width, width)
        tr_y = st.slider("Top-Right Y", 0, height, 0)
        br_x = st.slider("Bottom-Right X", 0, width, width)
        br_y = st.slider("Bottom-Right Y", 0, height, height)
        bl_x = st.slider("Bottom-Left X", 0, width, 0)
        bl_y = st.slider("Bottom-Left Y", 0, height, height)

    points = [(tl_x, tl_y), (tr_x, tr_y), (br_x, br_y), (bl_x, bl_y)]
    return points



def main():
    st.title("De-Warping of the color chart")
    if "chart_img" in st.session_state:
        st.title("Perspective correction of the color chart")
        st.markdown("In this page, you can correct the perspective of the color chart.")
        # st.markdown("You can also add padding to the image to avoid cropping.")
        st.markdown("After adjusting the perspective of the image, you can save the image to memory.")
        st.markdown("You can also go back to the previous page to separate the color chart segments.")

        is_color_chart_in_session_state()
        image = st.session_state["chart_img"]

        pts = select_perspective_points(image.shape)
        rotated_image = perspective_correction(image, pts)
        st.image(rotated_image, caption="Warp Image", use_column_width=True)



        if st.button("Save Images to memory"):
            st.session_state["chart_img"] = rotated_image
        switch_to_color()
    else:
        st.write("Please go to page 1 to upload the image and select the color chart")
        switch_to_cropping()

# Streamlit app execution
if __name__ == '__main__':
    main()