import streamlit as st
import cv2
import zipfile
from io import BytesIO
import numpy as np

# Function to resize image
def resize_image(image, scale_percent):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

# Function to process images
def process_images(uploaded_files, scale_percent):
    images_dict = {}
    for uploaded_file in uploaded_files:
        file_bytes = uploaded_file.read()
        image = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
        resized_image = resize_image(image, scale_percent)
        images_dict[uploaded_file.name] = resized_image
    return images_dict

# Streamlit UI
st.title("Image Resizer")
uploaded_files = st.file_uploader("Upload images", type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'], accept_multiple_files=True)
scale_percent = st.selectbox("Select resize percentage:", [25, 50, 75])

if st.button("Resize Images"):
    if uploaded_files:
        images_dict = process_images(uploaded_files, scale_percent)
        
        # Create a zip file
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
            for filename, image in images_dict.items():
                _, buffer = cv2.imencode('.jpg', image)
                zip_file.writestr(filename, buffer.tobytes())
        
        st.success("Images resized and zipped successfully!")
        st.download_button(
            label="Download ZIP",
            data=zip_buffer.getvalue(),
            file_name="resized_images.zip",
            mime="application/zip"
        )
    else:
        st.error("Please upload at least one image.")