import streamlit as st


st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.write("# Welcome to The app! 👋")

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
    
# Interactive coral color analysis

This project implements the segment anything algorithm, and asses the coral heatlh accoring to the color chart given.
The project not only requires segmentation, it also applies Optical Character Recognition  (OCR) , it manipulates the image to create a Custom Color Chart 
The combination of these techniques results in the mapping of colors over the coral of interest. 
You are in the welcome page so lets think step by step 


## Usage/Examples

Welcome to The app! 👋
This is a modular application , meaning each of the names you see on the side bar is a stage on the sequence of steps to get the result . There are 3 main stages:


### Step 1 - Cropping images 
In this part you will select two section of your images:
* the coral image
* the color chart 

After this you will save the images into the memory of the machine 

### Step 2 - Separate Colors from the chart
In this section you are required to select the color sections from the color chart (Up, Down, Left, Right)
Also, this assumes that you have a color chart as in the "Screenshot section".

The sequence is as follows :
* Select up, down ... , Try to get the letter as best a posible
* Process the color crops - this will try to correct the titl on the image to make the OCR easier
* Start the OCR by pressing "Build the Color Chart" button 
    - if the program can not detect 6 labels it will tell you 
    - you can reselect the area you have problems and process it again
* If the OCR is succesful it will deploy the custom color chart and you can save it as a PNG file and also as text file.

### Step 3 - Mapping the custom color chart
In this secction the we process the coral image and finaly analize the coral Colors

Start with the segmentation of the image of the coral using a finetunes the SAM model (Thanks Meta AI ) called "CoralSCOP" (Thanks to the authors of the paper)
* Press start segmentation button
* Select the Image of the coral that you want to analyze
* Trigger the analisis byt pressing "Analyze colors in the selected image" button 
* Wait for the results

#### Interpreting the Results 
 The firts result you will get is a quick analysis of the color we can find on the coral as a pie chart that will tell you the percentaje of colors on the image
- Then it wll show the color closer to a "Ideal color chart" - meaning it calculates the color closest to the color chart what we hardcoded with the correspoing RGB values from the color chart.
- Finally the las images will be an figure that shows the original image, a mapped image generated with the colors from the custom color chart and the percentage of each custon color on the picture 

## Image manipulation sections 
Since the color chart is a key part of the analysis, we have included a section to manipulate the image to create a custom color chart.
* `rotation of the color chart`:  You can rotate the image 180 degrees to the right or to the left
* `dewarp the image`:  You can fix the perspective of the image. This is useful when the image is not taken from the front and the OCR can not detect the labels
* `manual selection`: You can select the manually the colors one by one from the chart segments from the image (The last resort if the OCR can not detect the labels , is time comsuming but it works)

## Several images at the same time - Batch processing
If you have several images to analyze you can use the batch processing section.
* Load the color chart or select the custom color chart from a reprensentative image on the firts section
* Go to the batch processing section
* Select the images you want to analyze
* Press the button to start the batch processing
* Wait for the results

The results will be saved on a zip file that you can download and check the results. This process will take more time since it will process all coral detected on all the images you have selected.



## Badges

Add badges from somewhere like: [shields.io](https://shields.io/)

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)

## Screenshots

"""
)
# st.image("images/ColorChart.jpg", caption="Screenshot of the color chart", use_column_width=True)
# st.image("images/Figure_for_neus_paper-underwater-app.png", caption="Conceptual workflow", use_column_width=True)