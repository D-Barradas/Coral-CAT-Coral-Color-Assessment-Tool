import streamlit as st
import pandas as pd
import cv2
from streamlit_extras.image_selector import image_selector, show_selection
import numpy as np
from streamlit_extras.switch_page_button import switch_page


st.set_page_config(page_title="Cropping Image", page_icon="🌍")

# st.markdown("# Create a cropped")
# st.sidebar.header("Create crop")
# st.write(
#     """This Should crop and same the color from the color chart."""
# )

with open("load_functions.py") as f:
    exec(f.read())


def correct_tilt(image, gradient):

    # Load the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect edges in the image
    if gradient == False:
        t_lower = 50 # Lower Threshold 
        t_upper = 150 # Upper threshold 
        aperture_size = 3 # Aperture size 

        edges = cv2.Canny(gray, t_lower, t_upper, 
                apertureSize=aperture_size  )
        # Detect lines in the image using Hough Transform
        lines = cv2.HoughLines(edges, 1, np.pi/180, 100)
        
    elif gradient == True:
        t_lower = 100 # Lower Threshold 
        t_upper = 200 # Upper threshold 
        aperture_size = 5 # Aperture size 
        # print ( t_lower , t_upper , aperture_size)
        
        # Applying the Canny Edge filter with L2Gradient = True 
        #  # Min number of votes for valid line

        edges = cv2.Canny(gray, t_lower, t_upper, 
                apertureSize = aperture_size,  
                L2gradient = gradient ) 
        # Detect lines in the image using Hough Transform
        lines = cv2.HoughLines(edges, 1, np.pi/180, 50)
    
    # print(type(lines))
    
    # Calculate the angle of the lines
    angles = []
    for line in lines:
        rho, theta = line[0]
        angle = (theta * 180 / np.pi) - 90
        angles.append(angle)

    # Calculate the median angle
    median_angle = np.median(angles)
    
    # Rotate the image to correct the tilt
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    corrected_image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    # # Save the corrected image
    # corrected_image_path = 'corrected_' + image_path
    # cv2.imwrite(corrected_image_path, corrected_image)
    
    return corrected_image


def select_coral_and_color_chart_area(image):
    """
    Allows user to select two regions (coral, color_chart) using bounding boxes.
    Saves coordinates in a dictionary and displays the selected regions.
    """
    ## sort out how to get the coordinates on this dictionary
    selected_regions = {}
    # selection_type = st.radio("Selection mode:", ["box"], index=1, horizontal=True)
    selection_type = "box"

    selection = image_selector(image=image, selection_type=selection_type, key="separate")

    if selection:
        st.json(selection, expanded=False)
        try : 
            show_selection(image, selection)
        except ValueError as e:  #raised if `y` is empty.
            print (f"An error occurred while showing image selection: {e}")
            st.markdown("## Can't display image ")
            st.markdown("## Make the selection again, please! ")
            pass
        
        if selection_type == "box" : 
            # print (selection["selection"]["box"], type(selection["selection"]["box"]))
            if len(selection["selection"]["box"] ) != 0 :
                # print(selection["selection"]["box"][0]["x"])
                # print(selection["selection"]["box"][0]["y"])
                x = selection["selection"]["box"][0]["x"][0]
                y = selection["selection"]["box"][0]["y"][0]
                w = selection["selection"]["box"][0]["x"][1]
                h = selection["selection"]["box"][0]["y"][1]

                st.write(f"Box Coordinates: x={x}, y={y}, width={w}, height={h}")

                if st.button("Save Coral"):
                    st.session_state["coral"] = (x, y, w, h)
                if st.button("Save Color Chart"):
                    st.session_state["chart"] = (x, y, w, h)



                st.write("Saved Coordinates:")
                st.write("Coral:", st.session_state.get("coral", "Not saved"))
                st.write("Color Chart:", st.session_state.get("chart", "Not saved"))


                # print (len(st.session_state), type(st.session_state))

                # if len(st.session_state) == 3:
                if st.session_state["coral"] and st.session_state["chart"]:
                    st.write("Coral and Chart selected.")
                    selected_regions = { 
                        "coral":st.session_state["coral"],
                        "chart":st.session_state["chart"]

                        }
    return selected_regions

def get_coral_and_chart_image(local_image , boxes):

    dictionary_of_crops = {} 
    dictionary_of_greys = {} 

    for t in boxes.keys():
        print (t)

        if t == "chart":
            cropped_image = crop_my_image(image=local_image,boxes=boxes, tag=t)
            # try :
            #     cropped_image = correct_tilt(cropped_image,False)
            # except:
            #     print ("Could not correct the rotation normally; Applying the Canny Edge filter with L2Gradient = True")
            #     cropped_image = correct_tilt(cropped_image,True)
            dictionary_of_crops[t]= cropped_image
            dictionary_of_greys[t]= cropped_image

            # resized_gray, resized_cropped  = resize_image(cropped_image) 
            # dictionary_of_crops[t]= resized_cropped
            # dictionary_of_greys[t]= resized_gray
        else:
            cropped_image = crop_my_image(image=local_image,boxes=boxes, tag=t)
            dictionary_of_crops[t]= cropped_image
            dictionary_of_greys[t]= cropped_image
            # resized_gray, resized_cropped  = resize_image(cropped_image) 
            # dictionary_of_crops[t]= resized_cropped
            # dictionary_of_greys[t]= resized_gray

    # print (selected_regions)
    return dictionary_of_crops , dictionary_of_greys


def resize_image(crop_img):
    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    resized_gray= cv2.resize(gray,(2000,500) , interpolation = cv2.INTER_CUBIC)
    resized_cropped= cv2.resize(crop_img,(2000,500) , interpolation = cv2.INTER_CUBIC)

    # this didnt work
    # resized_gray= cv2.resize(gray,(1800, 1200 ) , interpolation = cv2.INTER_AREA)
    # resized_cropped= cv2.resize(crop_img,(1800,1200 ) ,interpolation = cv2.INTER_AREA)
    return resized_gray, resized_cropped


def switch_to_next():
    """Must be one of ['streamlit starting page', 'cropping color chart', 'separate color chart', 'analysis and mapping', 'rotation of the color chart']"""

    want_to_contribute = st.button("Go next phase?")
    if want_to_contribute:
        switch_page("separate color chart")



def main():
    # ---------- HEADER ----------

    # st.title('Image Segmentation and Analysis')
    st.title("Select the coral and the chart")

    # # initialize the session state
    # if "coral_img" not in st.session_state:
    #     st.session_state["coral"] = None
    # if "chart_img" not in st.session_state:
    #     st.session_state["coral"] = None
    if "chart" not in st.session_state:
        st.session_state["chart"] = None
    if "coral" not in st.session_state:
        st.session_state["coral"] = None

    if st.button("Reset Session"):
        for key in st.session_state.keys():
            del st.session_state[key]

        if "chart" not in st.session_state:
            st.session_state["chart"] = None
        if "coral" not in st.session_state:
            st.session_state["coral"] = None


    uploaded_file = st.file_uploader("Choose an image...", type=["bmp", "jpg", "jpeg", "png", "svg"], key="image")
    if uploaded_file is not None:
        image = get_image(uploaded_file)


        selected_regions = select_coral_and_color_chart_area(image)
        # print (selected_regions)
        if st.button("Save Images to Memory"):
            dictionary_of_crops, dictionary_of_greys = get_coral_and_chart_image(image , selected_regions)
            # check that dictionary has the right keys
            try:
                if "coral" not in dictionary_of_crops.keys() or "chart" not in dictionary_of_crops.keys():
                    st.write("Please reset the session and try again")
                    # st.stop()
                else : 
                    list_of_images=[item for key,item in dictionary_of_crops.items()]
                    st.image(list_of_images,use_column_width=True)
                    print(dictionary_of_crops.keys())
                    st.session_state["coral_img"] = dictionary_of_crops["coral"]
                    st.session_state["chart_img"] = dictionary_of_crops["chart"]
            except:
                st.write("Please reset the session and try again")
                # st.stop()
            # list_of_images=[item for key,item in dictionary_of_crops.items()]
            # st.image(list_of_images,use_column_width=True)
            # print(dictionary_of_crops.keys())
            # st.session_state["coral_img"] = dictionary_of_crops["coral"]
            # st.session_state["chart_img"] = dictionary_of_crops["chart"]

    
    st.markdown("## You can load your own color chart")
    st.write("Please upload a text file with the color chart in the following format:")
    st.write("color_name: (R, G, B)")
    st.write("For example:")
    st.write("B1: (234, 233, 228)")
    st.write("B2: (225, 222, 189)") 
    # st.write("B3: (225, 222, 158)")
    # st.write("B4: (219, 218, 110)")
    # st.write("B5: (190, 199, 83)")
    # st.write("B6: (157, 179, 72)")

    uploaded_file_txt = st.file_uploader("Choose a text file...", type=["txt"], key="color_chart")
    if uploaded_file_txt is not None:

        my_personal_chart  = pd.read_csv(uploaded_file_txt, sep=":", header=None)
        my_personal_chart.set_index(0, inplace=True)
        my_personal_chart = my_personal_chart[1].apply(lambda x: tuple(map(int, x.replace("(","").replace(")","").split(",")))).to_dict()
        st.session_state["custom_color_chart"] = my_personal_chart
        OcrAnalysis.plot_custom_colorchart(my_personal_chart)

   


    # go to the next page if done 
    switch_to_next()

# Streamlit app execution
if __name__ == '__main__':
    main()