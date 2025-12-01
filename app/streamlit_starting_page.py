import streamlit as st


st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.write("# Welcome to the Coral CAT: Coral Color Assessment Tool! 👋")

st.sidebar.success("Select each stage above, one by one.")

# # initialize all st.session_state variables
# variables = ["chart_img", "coral_img", "rotated_img", "custom_chart", "up", "down", "left", "right","custom_color_chart"]

# for cardinalities in variables:
#     if cardinalities not in st.session_state:
#         st.session_state[cardinalities] = None

# add a markdown section to explain that this button will reset the session
# st.markdown(
#     """
#     # Reset Session
#     This button will reset the session and clear all the images you have uploaded.
#     """)
# # allow the user to reset the session
# if st.button("Reset Session"):
#     for cardinalities in variables:
#         st.session_state[cardinalities] = None



    


st.markdown(
    """
# Welcome to the Coral CAT: Coral Color Assessment Tool

**Interactive coral color analysis**

Coral CAT is a semi-automatic image analysis tool that assigns color codes to coral images. The pipeline allows the standardized and automatic extraction of the color (RGB) of each pixel of a cropped coral image. It assigns the closest color code extracted from the picture's color chart. We combine the available tools, such as CoralSCOP and Optical Character Recognition (OCR).

## Usage

This is a modular application, meaning that each tab on the sidebar is a consecutive step. The workflow is divided into three main steps:

### Step 1 - Upload the image and define the color chart and coral fragment/colony areas

In this tab, you can upload your image and define the areas corresponding to the color chart and the coral fragment of the colony. The specified and selected areas must be saved in the memory before moving to the next step.

### Step 2 - Build a Custom Color Chart

In this section, you are required to select the color hue sections from the color chart (Up, Down, Left, and Right corresponding to B, D, E, and C, respectively). It is important to keep the color hues in the right position.

The sequence is as follows:
- Select and save the four color hue areas; try to get the letter as best as possible
- Process the color crops - this will try to correct the tilt on the image to make the OCR easier
- Start the OCR by clicking on the "Build the Color Chart" button
  - If the program cannot detect six labels, it will tell you
  - You can reselect the area where you have problems and process it again
- If the OCR is successful, it will show the custom color chart image, which can be saved as a PNG file and exported as a text file.

### Step 3 - Color analysis and mapping

In this section, the saved coral image area is segmented using a finetuned [SAM model](https://github.com/facebookresearch/segment-anything) called ["CoralSCOP"](https://github.com/zhengziqiang/CoralSCOP).  Please cite these amazing tools if you use them. 
- Press the start segmentation button.
- Select the image segment corresponding to the cropped coral area you want to analyze.
- Start the analysis by clicking the "Analyze colors in the selected image" button.

The results provided include a color analysis.

Finally, the last image will be a figure showing the original image, a mapped image generated with the colors from the custom color chart, and the percentage of each color on the picture.

## Image correction sections

Since the color chart is a key part of the analysis, we have included a section to manipulate the image to create a custom color chart.
- Rotation of the color chart: You can rotate the image 180 degrees to the right or the left
- Dewarp the image: You can fix the image's perspective. This is useful when the OCR cannot detect the labels
- Manual selection: You can select the colors one by one from the chart segments from the image (in case the OCR cannot detect the labels)

## Batch mode

This option allows the analysis of several images simultaneously using the same color chart.

## Multi-segment analysis

This option allows the analysis of several segments at the same time.
"""
)
st.image("images/Coral-Cat_app.png", caption="App usage", use_column_width=True)
