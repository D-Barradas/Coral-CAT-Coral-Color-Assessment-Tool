import streamlit as st
# from PIL import Image, UnidentifiedImageError
from io import BytesIO
# import requests
# import os
import cv2
# from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

from streamlit_extras.image_selector import image_selector, show_selection
from streamlit_extras.switch_page_button import switch_page
import numpy as np

import easyocr , os , ssl
ssl._create_default_https_context = ssl._create_unverified_context


st.set_page_config(page_title="Separate Color from the Chart", page_icon="📊")

st.markdown("# Separate Color from the Chart")
st.sidebar.header("Select color chart regions")

st.write(
    """ The color chart is on different rotations This will produce a custom color chart from the selection """
)
with open("load_functions.py") as f:
    exec(f.read())


def define_region_selection(image):
    """
    Allows user to select four regions (Up, Down, Left, Right) using bounding boxes.
    Saves coordinates in a dictionary and displays the selected regions.
    """
    ## sort out how to get the coordinates on this dictionary
    selected_regions = {}
    # selection_type = st.radio("Select a selection method:", ["box"], index=0, horizontal=True)

    #convert the numpy.ndarray image to pill
    # image = Image.fromarray(image)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    selection_type = "box"
    selection = image_selector(image=image, selection_type=selection_type, key="Color_chart")

    if selection:
        st.json(selection, expanded=False)
        show_selection(image, selection)
        if selection_type == "box" : 
            if len(selection["selection"]["box"] ) != 0 :
                # print (len(selection["selection"]["box"]))

                # print(selection["selection"]["box"][0]["x"])
                # print(selection["selection"]["box"][0]["y"])
                x = selection["selection"]["box"][0]["x"][0]
                y = selection["selection"]["box"][0]["y"][0]
                w = selection["selection"]["box"][0]["x"][1]
                h = selection["selection"]["box"][0]["y"][1]

                st.write(f"Box Coordinates: x={x}, y={y}, width={w}, height={h}")

                if st.button("Save Up"):
                    st.session_state["up"] = (x, y, w, h)
                if st.button("Save Down"):
                    st.session_state["down"] = (x, y, w, h)

                if st.button("Save Left"):
                    st.session_state["left"] = (x, y, w, h)

                if st.button("Save Right"):
                    st.session_state["right"] = (x, y, w, h)


                st.write("Saved Coordinates:")
                st.write("Up:", st.session_state.get("up", "Not saved"))
                st.write("Down:", st.session_state.get("down", "Not saved"))
                st.write("Left:", st.session_state.get("left", "Not saved"))
                st.write("Right:", st.session_state.get("right", "Not saved"))

                # print (len(st.session_state), type(st.session_state))

                # if len(st.session_state) == 5:
                # st.write("The session state has 4 regions selected.")
                selected_regions = { 
                    "up":st.session_state["up"],
                    "down":st.session_state["down"],
                    "left":st.session_state["left"],
                    "rigth":st.session_state["right"]
                }

    # print (selected_regions)
    return selected_regions


def resize_image(crop_img):
    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    resized_gray= cv2.resize(gray,(2000,500) , interpolation = cv2.INTER_CUBIC)
    resized_cropped= cv2.resize(crop_img,(2000,500) , interpolation = cv2.INTER_CUBIC)
    return resized_gray, resized_cropped

def resize_and_rotate_image(local_image, boxes):
    # local_image = dewarped_image
    dictionary_of_crops = {} 
    dictionary_of_greys = {} 

    for t in ["up","down","left","rigth"]:

        if t == "down":
            cropped_image = crop_my_image(image=local_image,boxes=boxes, tag=t)
            cropped_image = cv2.rotate(cropped_image, cv2.ROTATE_180  )

            # try :
            #     cropped_image = correct_tilt(cropped_image,False)
            # except:
            #     print ("Could not correct the rotation normally; Applying the Canny Edge filter with L2Gradient = True")
            #     cropped_image = correct_tilt(cropped_image,True)
            
            resized_gray, resized_cropped  = resize_image(cropped_image) 
            dictionary_of_crops[t]= resized_cropped
            dictionary_of_greys[t]= resized_gray
            # cv2.imwrite(path_name, cropped_image)
        elif t == "left":
            # cropped_image = crop_my_image(image=local_image,tag=t)
            cropped_image = crop_my_image(image=local_image,boxes=boxes, tag=t)
            cropped_image = cv2.rotate(cropped_image, cv2.ROTATE_90_CLOCKWISE ) 

            # try :
            #     cropped_image = correct_tilt(cropped_image,False)
            # except:
            #     print ("Could not correct the rotation normally; Applying the Canny Edge filter with L2Gradient = True")
            #     cropped_image = correct_tilt(cropped_image,True)

            resized_gray, resized_cropped  = resize_image(cropped_image) 
            dictionary_of_crops[t]= resized_cropped
            dictionary_of_greys[t]= resized_gray

            # cv2.imwrite(path_name, cropped_image)
        elif t == "rigth":
            # cropped_image = crop_my_image(image=local_image,tag=t)
            cropped_image = crop_my_image(image=local_image,boxes=boxes, tag=t)
            cropped_image = cv2.rotate(cropped_image, cv2.ROTATE_90_COUNTERCLOCKWISE ) 

            # try :
            #     cropped_image = correct_tilt(cropped_image,False)
            # except:
            #     print ("Could not correct the rotation normally; Applying the Canny Edge filter with L2Gradient = True")
            #     cropped_image = correct_tilt(cropped_image,True)

            resized_gray, resized_cropped  = resize_image(cropped_image) 
            dictionary_of_crops[t]= resized_cropped
            dictionary_of_greys[t]= resized_gray
            
            # cv2.imwrite(path_name, cropped_image)
        else:
            # cropped_image = crop_my_image(image=local_image,tag=t)
            cropped_image = crop_my_image(image=local_image,boxes=boxes, tag=t)
            resized_gray, resized_cropped  = resize_image(cropped_image)

            # try :
            #     cropped_image = correct_tilt(cropped_image,False)
            # except:
            #     print ("Could not correct the rotation normally; Applying the Canny Edge filter with L2Gradient = True")
            #     cropped_image = correct_tilt(cropped_image,True)
 
            dictionary_of_crops[t]= resized_cropped
            dictionary_of_greys[t]= resized_gray
            # cv2.imwrite(path_name, cropped_image)
    return dictionary_of_crops, dictionary_of_greys

def apply_correct_tilt (dictionary_of_crops):
    dictionary_of_crops_corrected = {}
    for t in ["up","down","left","rigth"]:
        cropped_image = dictionary_of_crops[t]
        try :
            cropped_image = correct_tilt(cropped_image,False)
        except:
            print ("Could not correct the rotation normally; Applying the Canny Edge filter with L2Gradient = True")
            cropped_image = correct_tilt(cropped_image,True)
        dictionary_of_crops_corrected[t]= cropped_image

    return dictionary_of_crops_corrected

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


custom_chart_key_code_and_order = {
    0:['B1', 'B2', 'B3', 'B4', 'B5', 'B6'],
    1:['D1', 'D2', 'D3', 'D4', 'D5', 'D6'],
    2:['E1', 'E2', 'E3', 'E4', 'E5', 'E6'], 
    3:['C1', 'C2', 'C3', 'C4', 'C5', 'C6']
    }

color_map_RGB = {'Black':(0,0,0),'White':(255,255,255),'B1': (247, 248, 232),'B2': (243, 244, 192),'B3': (234, 235, 137),'B4': (200, 206, 57),'B5': (148, 157, 56),
                    'B6': (92, 116, 52),'C1': (247, 235, 232),'C2': (246, 201, 192),'C3': (240, 156, 136),'C4': (207, 90, 58),'C5': (155, 50, 32),'C6': (101, 27, 13),
                    'D1': (246, 235, 224),'D2': (246, 219, 191),'D3': (239, 188, 135),'D4': (211, 147, 78),'D5': (151, 89, 36),'D6': (106, 58, 22),'E1': (247, 242, 227),
                    'E2': (246, 232, 191),'E3': (240, 213, 136),'E4': (209, 174, 68),'E5': (155, 124, 45),'E6': (111, 85, 34)}


def calculate_distance(bbox1, bbox2):
    """Calculates the distance between two bounding boxes.

    Args:
        bbox1: A NumPy array representing the coordinates of the first bounding box.
        bbox2: A NumPy array representing the coordinates of the second bounding box.

    Returns:
        The distance between the two bounding boxes.
    """

    center_x1, center_y1 = (bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2
    center_x2, center_y2 = (bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2

    distance = np.sqrt((center_x1 - center_x2) ** 2 + (center_y1 - center_y2) ** 2)

    return distance



def find_optimal_placement(bboxes, new_bbox_area):
    """Finds the optimal placement for a new bounding box based on distances to existing bounding boxes.

    Args:
        bboxes: A list of existing bounding boxes as NumPy arrays.
        new_bbox_area: The desired area of the new bounding box.

    Returns:
        A tuple representing the coordinates of the optimal placement for the new bounding box.
    """

    # Calculate the average distance between existing bounding boxes
    avg_distance = 0
    for i in range(len(bboxes)):
        for j in range(i + 1, len(bboxes)):
            idx_rest = j -i  
            if idx_rest == 1 :
                avg_distance += calculate_distance(bboxes[i], bboxes[j])
                distance = calculate_distance(bboxes[i], bboxes[j])
                print(f"Distance between bounding boxes {i} and {j}: {distance} inside optimal placement")
    # avg_distance /= len(bboxes) * (len(bboxes) - 1) / 2
    print( avg_distance, len(bboxes) ,   len(bboxes) - 1 )
    # avg_distance /= len(bboxes) * (len(bboxes) - 1)
    avg_distance /= len(bboxes)


    # Find the bounding box with the largest area
    largest_area_bbox = max(bboxes, key=lambda bbox: (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))

    print(f"avg_distance:{avg_distance}")

    # Calculate the desired width and height for the new bounding box
    new_bbox_width = np.sqrt(new_bbox_area)
    new_bbox_height = new_bbox_width

    # Iterate through different placements and find the one with the smallest average distance to existing bounding boxes
    optimal_placement = None
    min_avg_distance = float('inf')
    for x in range(0, 1000, int(avg_distance / 2)):
        for y in range(0, 1000, int(avg_distance / 2)):
            new_bbox = np.array([x, y, x + new_bbox_width, y + new_bbox_height])
            avg_distance_to_existing = 0
            for existing_bbox in bboxes:
                avg_distance_to_existing += calculate_distance(new_bbox, existing_bbox)
            avg_distance_to_existing /= len(bboxes)
            if avg_distance_to_existing < min_avg_distance:
                optimal_placement = new_bbox
                min_avg_distance = avg_distance_to_existing

    return optimal_placement


def place_copy(bounding_boxes):
    """Places a copy of the first bounding box at the average distance of the last bounding box, adding the distance to the x-axis.

  Args:
    bounding_boxes: A list of bounding boxes, each represented as a NumPy array of shape (4,).

  Returns:
    A list of bounding boxes with the added copy.
    """

    # Calculate the average distance between the last bounding box and the others
    # avg_distance = 0
    # for bbox in bounding_boxes[:-1]:
    #     avg_distance += np.linalg.norm(bbox[:2] - bounding_boxes[-1][:2])
    #     avg_distance /= len(bounding_boxes) - 1
        # Calculate the average distance between existing bounding boxes
    avg_distance = 0
    for i in range(len(bounding_boxes)):
        for j in range(i + 1, len(bounding_boxes)):
            idx_rest = j -i  
            if idx_rest == 1 :
                avg_distance += calculate_distance(bounding_boxes[i], bounding_boxes[j])
                distance = calculate_distance(bounding_boxes[i], bounding_boxes[j])
                print(f"Distance between bounding boxes {i} and {j}: {distance} inside place copy")
    # avg_distance /= len(bboxes) * (len(bboxes) - 1) / 2
    # print( avg_distance, len(bounding_boxes) ,   len(bounding_boxes) - 1 )
    # avg_distance /= len(bboxes) * (len(bboxes) - 1)
    avg_distance /= len(bounding_boxes)



    # print(avg_distance)
    # Create a copy of the first bounding box
    copied_bbox = bounding_boxes[-1].copy()

    # Add the average distance to the x-coordinate of the copied bounding box
    copied_bbox[0] += avg_distance
    copied_bbox[2] += avg_distance

    # Append the copied bounding box to the list
    bounding_boxes.append(copied_bbox)

    return bounding_boxes

def switch_to_cropping():
    """Must be one of ['streamlit starting page', 'cropping color chart', 'separate color chart', 'analysis and mapping', 'rotation of the color chart']"""

    want_to_contribute = st.button("Upload the image?")
    if want_to_contribute:
        switch_page("cropping color chart")

def switch_to_rotation_page():
    """Must be one of ['streamlit starting page', 'cropping color chart', 'separate color chart', 'analysis and mapping', 'rotation of the color chart']"""

    want_to_contribute = st.button("Rotate the image?")
    if want_to_contribute:
        switch_page("rotation of the color chart")

def switch_to_next():
    """Must be one of ['streamlit starting page', 'cropping color chart', 'separate color chart', 'analysis and mapping', 'rotation of the color chart']"""

    want_to_contribute = st.button("Go next phase?")
    if want_to_contribute:
        switch_page("analysis and mapping")

def switch_to_manual():
    """Must be one of ['streamlit starting page', 'cropping color chart', 'separate color chart', 'analysis and mapping', 'rotation of the color chart']"""

    want_to_contribute = st.button("Manual selection?")
    if want_to_contribute:
        switch_page("manual selection of colors")

def switch_to_perspective():
    """Must be one of ['streamlit starting page', 'cropping color chart', 'separate color chart', 'analysis and mapping', 'rotation of the color chart']"""

    want_to_contribute = st.button("Modify the perpective?")
    if want_to_contribute:
        switch_page("dewarp the image")



def main():
    if "chart_img" not in st.session_state:

        st.write("Please go to page 1 to upload the image")
        switch_to_cropping()
    else:
        st.write("Color chart image is already in session state")

        image = st.session_state["chart_img"]
        for cardinalities in ["up", "down", "left" ,"right" ]:
            if cardinalities not in st.session_state:
                st.session_state[cardinalities] = []


        switch_to_rotation_page()
        switch_to_perspective()
        # switch_to_manual()
        selected_regions = define_region_selection(image)

        list_of_images =[]
        if st.button("process crops"):
            dictionary_of_crops, dictionary_of_greys = resize_and_rotate_image(local_image=image,boxes=selected_regions)
            list_of_images=[item for key,item in dictionary_of_crops.items()]
            list_of_images_grey=[item for key,item in dictionary_of_greys.items()]
            
            st.image(list_of_images)
            # st.image(list_of_images_grey)
            st.session_state["color_crops"] = list_of_images
            st.session_state["grey_crops"] = list_of_images_grey
        
        if st.button("Apply tilt correction"):
            dictionary_of_crops, dictionary_of_greys = resize_and_rotate_image(local_image=image,boxes=selected_regions)

            # this is the dictionary of crops in colors
            dictionary_of_crops = apply_correct_tilt(dictionary_of_crops)
            list_of_images=[item for key,item in dictionary_of_crops.items()]
            st.image(list_of_images)
            st.session_state["color_crops"] = list_of_images
            # this is the dictionary of crops in grey
            # dictionary_of_greys = apply_correct_tilt(dictionary_of_greys)
            # list_of_images_grey=[item for key,item in dictionary_of_greys.items()]
            # st.session_state["grey_crops"] = list_of_images_grey

        if st.button("Build the Color Chart"):
            reader = easyocr.Reader(["en"],gpu=True)
            my_personal_chart = {}
            my_dictionary_to_check_number_of_boxes = {}
            for index , color_chart_segment in enumerate ( st.session_state.get("color_crops"))  :
                result = reader.readtext(color_chart_segment)
                bboxes, text_list = OcrAnalysis.get_bounding_boxes(result)
                if len(bboxes) == 6  :
                    text_list = custom_chart_key_code_and_order[index]
                    my_dictionary_to_check_number_of_boxes[index]  = {"bboxes":bboxes, 
                                                                    "text_list": text_list, 
                                                                    #   "color_chart_segment":color_chart_segment, 
                                                                    "num_of_boxes":len(bboxes)}
            try :
                # check that my_dictionary_to_check_number_of_boxes has 4 keys
                if len(my_dictionary_to_check_number_of_boxes) == 4:
                    for index , color_chart_segment in enumerate ( st.session_state.get("color_crops"))  :
                        bboxes, text_list = my_dictionary_to_check_number_of_boxes[index]["bboxes"], my_dictionary_to_check_number_of_boxes[index]["text_list"] 
                        for t,bbox in zip(text_list,bboxes):
                            cropped_colors = OcrAnalysis.get_pixels_above_bbox(bbox=bbox,image=color_chart_segment)
                            df_color = get_colors_df(image=cropped_colors, number_of_colors=1, show_chart=False)
                            my_personal_chart[t]=tuple( round(x) for x in df_color["rgb_colors"][0].tolist() )

                    my_personal_chart["Black"] = tuple([0,0,0])
                    my_personal_chart["White"] = tuple([255,255,255])

                    st.session_state["custom_color_chart"] = my_personal_chart
                    OcrAnalysis.plot_custom_colorchart(my_personal_chart)
                else:
                    dictionary_size = len(my_dictionary_to_check_number_of_boxes.keys())
                    st.write(f"Warning: Only {dictionary_size} color segments with 6 boxes detected")
                    # from my_dictionary_to_check_number_of_boxes get the index of the missing color chart segment
                    # and then use the bboxes to fill the gap
                    missing = [key for key in range(4) if key not in my_dictionary_to_check_number_of_boxes.keys()]
                    exising = [key for key in range(4) if key in my_dictionary_to_check_number_of_boxes.keys()]
                    for index in missing:
                        bboxes = my_dictionary_to_check_number_of_boxes[exising[0]]["bboxes"]
                        text_list = custom_chart_key_code_and_order[index]
                        my_dictionary_to_check_number_of_boxes[index]  = {"bboxes":bboxes, 
                                                                    "text_list": text_list, 
                                                                    #   "color_chart_segment":color_chart_segment, 
                                                                    "num_of_boxes":len(bboxes)}
                    # and now we can build the color chart
                    for index , color_chart_segment in enumerate ( st.session_state.get("color_crops"))  :
                        bboxes, text_list = my_dictionary_to_check_number_of_boxes[index]["bboxes"], my_dictionary_to_check_number_of_boxes[index]["text_list"] 
                        for t,bbox in zip(text_list,bboxes):
                            cropped_colors = OcrAnalysis.get_pixels_above_bbox(bbox=bbox,image=color_chart_segment)
                            df_color = get_colors_df(image=cropped_colors, number_of_colors=1, show_chart=False)
                            my_personal_chart[t]=tuple( round(x) for x in df_color["rgb_colors"][0].tolist() )

                    my_personal_chart["Black"] = tuple([0,0,0])
                    my_personal_chart["White"] = tuple([255,255,255])

                    st.session_state["custom_color_chart"] = my_personal_chart
                    OcrAnalysis.plot_custom_colorchart(my_personal_chart)

            except:
                st.write ("Not sufficient boxes detected")
                st.write ("Switching to the manual selection page")
                switch_to_manual()

        if st.button("Save the color chart for later?"):
            # safe my_personal_chart to a file
            my_personal_chart = st.session_state["custom_color_chart"] 

            color_chart_str = "\n".join([f"{key}: {value}" for key, value in my_personal_chart.items()])

            # Create a BytesIO object and write the string to it
            color_chart_bytes = BytesIO()
            color_chart_bytes.write(color_chart_str.encode())
            color_chart_bytes.seek(0)

            st.write("Color chart saved to custom_color_chart.txt")
            # Download the file
            st.download_button(
                label="Download the color chart",
                data=color_chart_bytes,
                file_name="custom_color_chart.txt",
                mime="text/plain",
                key="download_button_color_chart",
            )
            # os.remove("custom_color_chart.txt")
     
            # download the file
            # st.download_button(
            #     label="Download the color chart",
            #     data= ,
            #     file_name="custom_color_chart.txt",
            #     mime="text/plain",
            #     key="download_button_color_chart",
            # )
            

        # if st.button("Detect Writing"):
        #     reader = easyocr.Reader(["en"],gpu=True) # this needs to run only once to load the model into memory
        #     my_personal_chart = {}
        #     # print (st.session_state.get("color_crops"))
        #     for index , color_chart_segment in enumerate ( st.session_state.get("color_crops"))  :
                
        #         result = reader.readtext(color_chart_segment)
        #         bboxes, text_list = OcrAnalysis.get_bounding_boxes(result)
        #         # print (len(bboxes), text_list)
        #         ## here there should be a check
        #         # if len(bboxes) != 6 :
        #         # check the index number of the color_chart_segment missing 
        #         # match with the new keys and fill the gap 
        #         # we could fill the gap with the ideal color chart since is the usual dark colors that are missing


        #         if len(bboxes) == 6  : ## meaning is equal to 6
        #             ## replace the text_list with the custom_chart_key_code_and_order 
        #             text_list = custom_chart_key_code_and_order[index]
        #         else :
        #             st.write(f"Warning can't detect all writting for : {custom_chart_key_code_and_order[index]} \n Trying with the grayscale")
        #             # list_of_images_grey[index]
        #             result = reader.readtext(st.session_state["grey_crops"][index])
        #             bboxes, text_list = OcrAnalysis.get_bounding_boxes(result)

        #             ## this is for debugging
        #             # print (len(bboxes), text_list, "Trying with the grayscale")
        #             # print(bboxes[0][2] ,bboxes[0][0], bboxes[0][3], bboxes[0][1])
        #             # for i in range(len(bboxes)):
        #             #     for j in range(i + 1, len(bboxes)):
        #             #         idx_rest = j -i  
        #             #         if idx_rest == 1 :
        #             #             distance = calculate_distance(bboxes[i], bboxes[j])
        #             #             print(f"Distance between bounding boxes {i} and {j}: {distance}")

        #             # Place a box next to the last 
        #             bboxes = place_copy(bounding_boxes=bboxes)
        #             text_list.append ("FAKE")
        #             text_list = custom_chart_key_code_and_order[index]




        




        #             # # Calculate the area of one of the existing bounding boxes
        #             # existing_bbox_area = (bboxes[0][2] - bboxes[0][0]) * (bboxes[0][3] - bboxes[0][1])

        #             # # Find the optimal placement for the new bounding box
        #             # new_bbox = find_optimal_placement(bboxes, existing_bbox_area)

        #             # # # Add the new bounding box to the list of existing bounding boxes
        #             # bboxes.append(new_bbox)
        #             # text_list.append ("A0")

        #             # # Print the bounding boxes
        #             # for bbox in bboxes:
        #             #     print(bbox)


        #         for t,bbox in zip(text_list,bboxes): 
        #             # print (t, bbox,"zip")
        #             cropped_colors = OcrAnalysis.get_pixels_above_bbox(bbox=bbox,image=color_chart_segment)
        #             df_color = get_colors_df(image=cropped_colors, number_of_colors=1, show_chart=False)
        #             # print (t,df_color["rgb_colors"][0].tolist())
        #             my_personal_chart[t]=tuple( round(x) for x in df_color["rgb_colors"][0].tolist() )

        #     # print (my_personal_chart.keys(), len(my_personal_chart.keys()))
        #     if my_personal_chart:

        #         new_keys = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6',
        #                     'D1', 'D2', 'D3', 'D4', 'D5', 'D6',
        #                     'E1', 'E2', 'E3', 'E4', 'E5', 'E6', 
        #                     'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 
        #                      ]
        #         # print (len(new_keys) , len(my_personal_chart.keys())) 
        #         if len(new_keys) == len(my_personal_chart.keys()):
        #             my_personal_chart = {new_keys[i]: my_personal_chart[old_key] for i, old_key in enumerate(my_personal_chart)}
        #             my_personal_chart["Black"] = tuple([0,0,0])
        #             my_personal_chart["White"] = tuple([255,255,255])
        #             st.session_state["custom_color_chart"] = my_personal_chart
        #             OcrAnalysis.plot_custom_colorchart(my_personal_chart)
        #         else :
        #             st.write("custom chart not complete")
        #             #print (f"not complete:{f_name}")
        #             # incomplete_to_fill.append(f_name)


        # go to the next page if done 
        switch_to_next()


# Streamlit app execution
if __name__ == '__main__':
    main()
