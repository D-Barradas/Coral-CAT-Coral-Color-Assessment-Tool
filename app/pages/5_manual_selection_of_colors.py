import streamlit as st
from streamlit_extras.image_selector import image_selector, show_selection
from streamlit_extras.switch_page_button import switch_page


with open("load_functions.py") as f:
    exec(f.read())

# Function to draw bounding boxes on the image
# def draw_bounding_boxes(image, boxes):
#     for box in boxes:
#         cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
#     return image
# print(st.session_state.get(" st.session_state["color_crops"]"))

## st.session_state["color_crops"] is the list of crops 

# for key in st.session_state.keys():
#     print ( key ) 

# print (st.session_state["color_crops"][0])
if st.button("Reset Session"):
    del st.session_state["bounding_boxes"]
# st.image(st.session_state["color_crops"][0])

# Initialize session state for bounding boxes if not already present
if 'bounding_boxes' not in st.session_state:
    st.session_state['bounding_boxes'] = {'up': [], 'down': [], 'left': [], 'right': []}


def switch_to_next():
    """Must be one of ['streamlit starting page', 'cropping color chart', 'separate color chart', 'analysis and mapping', 'rotation of the color chart']"""

    want_to_contribute = st.button("Go next phase?")
    if want_to_contribute:
        switch_page("analysis and mapping")


def create_the_custom_color_chart_locally(bboxes,color_chart_segment,image_option,dictionary):
    
    new_keys = {"up":   ["B1", "B2", "B3", "B4", "B5", "B6"],
                "down": ["D1", "D2", "D3", "D4", "D5", "D6"],
                "left": ["E1", "E2", "E3", "E4", "E5", "E6"], 
                "right":["C1", "C2", "C3", "C4", "C5", "C6"]
    }
    
    for t,bbox in zip(new_keys[image_option],bboxes): 
        x,y,w,h = bbox
        new_bbox = tuple ([int(x),int(y),int(w),int(h) ]) 
        #print (t, bbox,"here")
        cropped_colors = OcrAnalysis.get_pixels_above_bbox(bbox=new_bbox,image=color_chart_segment)
        df_color = get_colors_df(image=cropped_colors, number_of_colors=1, show_chart=False)

        # print (t,df_color["rgb_colors"][0].tolist())

        dictionary[t]=tuple( round(x) for x in df_color["rgb_colors"][0].tolist() )

    
    # my_personal_chart["Black"] = tuple([0,0,0])
    # my_personal_chart["White"] = tuple([255,255,255])
    # OcrAnalysis.plot_custom_colorchart(my_personal_chart)
    # st.session_state["custom_color_chart"] = my_personal_chart



def select_image():
    # Select image from session state
    image_option = st.selectbox("Select an image", ["up", "down", "left", "right"])
    print (image_option)
    selection_dictionary = {"up":0,"down":1,"left":2,"right":3}
    print (selection_dictionary[image_option]) 
    image = st.session_state["color_crops"][selection_dictionary[image_option]]
    # image = st.session_state["chart_img"]
    return image, image_option

def show_and_draw_boundingboxes(image, image_option):
# if image:
    # st.image(image, caption=f"Selected Image: {image_option}", use_column_width=True)
    st.image(image, caption=f"Selected Image: {image_option}", use_column_width=True)
    print(st.session_state['bounding_boxes'])

    
    # Allow user to draw bounding boxes
    st.write("Draw up to 6 bounding boxes on the image")
    if len(st.session_state['bounding_boxes'][image_option]) < 7:

        selection_type = "box"
        selection = image_selector(image=image, selection_type=selection_type, key="Color_chart_2")

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
                    if st.button("Add Bounding Box"):
                        st.session_state["bounding_boxes"][image_option].append((x, y, w, h))

                    st.write("Saved Coordinates:")
                    st.write("boxes:", st.session_state.get("bounding_boxes", "Not saved"))
                    # st.write("Down:", st.session_state.get("down", "Not saved"))
                    # st.write("Left:", st.session_state.get("left", "Not saved"))
                    # st.write("Right:", st.session_state.get("right", "Not saved"))

                    # if st.button("Save Up"):
                    #     st.session_state["up"] = (x, y, w, h)
                    # if st.button("Save Down"):
                    #     st.session_state["down"] = (x, y, w, h)

                    # if st.button("Save Left"):
                    #     st.session_state["left"] = (x, y, w, h)

                    # if st.button("Save Right"):
                    #     st.session_state["right"] = (x, y, w, h)
        
        # if st.button("Add Bounding Box"):
        #     st.session_state['bounding_boxes'][image_option].append((x1, y1, x2, y2))
    



    # # Display image with bounding boxes
    # image_with_boxes = draw_bounding_boxes(np.array(image), st.session_state['bounding_boxes'][image_option])
    # st.image(image_with_boxes, caption="Image with Bounding Boxes", use_column_width=True)
    
    # # Display selected colors
    # st.write("Selected Colors:")
    # for box in st.session_state['bounding_boxes'][image_option]:
    #     cropped_image = image.crop(box)
    #     avg_color = np.array(cropped_image).mean(axis=(0, 1))
    #     st.color_picker(f"Color from box {box}", value=f"#{int(avg_color[0]):02x}{int(avg_color[1]):02x}{int(avg_color[2]):02x}")

def switch_to_cropping():
    """Must be one of ['streamlit starting page', 'cropping color chart', 'separate color chart', 'analysis and mapping', 'rotation of the color chart']"""

    want_to_contribute = st.button("Upload the image?")
    if want_to_contribute:
        switch_page("cropping color chart")

# else:
#     st.write("No image found in the selected session state.")






def main():
    st.title("Manual Selection of Colors")
    if "color_crops" in st.session_state:
        image, image_option = select_image()
        show_and_draw_boundingboxes(image ,image_option)
        if st.button("Create Custom Color Chart"):
            my_personal_chart = {}

            for idx, option in enumerate ( st.session_state["bounding_boxes"].keys() ):
                print(idx,option)
                bboxes = st.session_state["bounding_boxes"][option]
                color_chart_segment = st.session_state["color_crops"][idx]
                print (color_chart_segment.shape)
                create_the_custom_color_chart_locally(bboxes=bboxes,color_chart_segment=color_chart_segment,image_option=option,dictionary=my_personal_chart)

            my_personal_chart["Black"] = tuple([0,0,0])
            my_personal_chart["White"] = tuple([255,255,255])
            OcrAnalysis.plot_custom_colorchart(my_personal_chart)
            st.session_state["custom_color_chart"] = my_personal_chart

        switch_to_next()
    else:
        st.write("No image found in the selected session state.")
        st.write("Please go to page 1 to upload the image and select the color chart")
        switch_to_cropping()



# Streamlit app execution
if __name__ == '__main__':
    main()
