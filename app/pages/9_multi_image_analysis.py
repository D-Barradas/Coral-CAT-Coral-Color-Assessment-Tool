import streamlit as st
import matplotlib.pyplot as plt
from streamlit_extras.switch_page_button import switch_page
import sys 
sys.path.append('../')
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from PIL import Image
import io


with open("load_functions.py") as f:
    exec(f.read())


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

def switch_to_manual():
    """Must be one of ['streamlit starting page', 'cropping color chart', 'separate color chart', 'analysis and mapping', 'rotation of the color chart']"""

    want_to_contribute = st.button("Go back to manual selection of colors?")
    if want_to_contribute:
        switch_page("manual selection of colors")

# if the color chart is not on session state ask the user to go to page 2 or 5  st.session_state["custom_color_chart"] 
def is_color_chart_in_session_state():
    if "custom_color_chart" not in st.session_state:

        st.write("Please go to page 2 or 5 to upload the color chart image")
        switch_to_color()
        switch_to_manual()

    else:
        st.write("Color chart image is already in session state")
        OcrAnalysis.plot_custom_colorchart(st.session_state["custom_color_chart"])


def is_cuda_available():
    """Checks if CUDA is available and can be used by PyTorch.

    Returns:
        bool: True if CUDA is available, False otherwise.
    """

    return torch.cuda.is_available()

# Function to load a model based on selection
def load_model_and_segment(image, model_option='Model_B'):
    sam_checkpoint = "../checkpoints/vit_b_coralscop.pth"  # this is coralSCOPE
    model_type = "vit_b"

    if is_cuda_available():
        st.markdown("CUDA is available!")
        device = torch.device("cuda:1")  # reactivate the previous line for the app
    else:
        st.markdown("CUDA is not available. Using CPU.")
        device = torch.device("cpu")

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    # Parameters for CoralScope
    points_per_side = 32
    pred_iou_thresh = 0.72
    stability_score_thresh = 0.62

    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=points_per_side,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,  # Requires open-cv to run post-processing
    )
    masks = mask_generator.generate(image)
    return masks


def multiple_mask_output(image):
    title= f'Image stacked'
    top_RGB_colors = get_colors(image, 6, True)
    color_keys_selected, color_selected_distance, lower_y_limit, higher_y_limit, hex_colors_map = calculate_distances_to_colors(image=image)
    plot_compare(image, color_keys_selected, color_selected_distance, lower_y_limit, higher_y_limit, hex_colors_map, title)
    plot_compare_mapped_image(image, st.session_state['custom_color_chart'])


    # Define a function to be called when the model selection changes
def on_model_change():
    # Update the session state to indicate that the model has changed
    st.session_state.model_changed = True
    st.session_state['segment'] = False

# Function to create and store the plot in session state
def create_plot(image, masks):
    plt.figure(figsize=(20,20))
    plt.imshow(image)
    show_anns(masks)
    plt.axis('on')

    # Adjust layout parameters to minimize white space
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Convert plot to an image and store in session state
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.0)
    buf.seek(0)
    st.session_state['plot_image'] = Image.open(buf)



# Initialize the session state
if 'model_changed' not in st.session_state:
    st.session_state['model_changed'] = False
    st.session_state['segment'] = False

def main():
    st.title("Multi Mode")

    if "coral_img" not in st.session_state or "custom_color_chart" not in st.session_state:
        st.write("Please go to page 1 to upload the image and page 2 to create the custom color chart")
        switch_to_cropping()

    else:
        st.write("Coral image and custom color chart are already in session state")
    # start with the image 

        coral_image = st.session_state["coral_img"]

        color_chart = st.session_state["custom_color_chart"]
        # Display the image
        st.image(coral_image, caption='Coral Image', use_column_width=True)

    # then the user can select the model

        # Model selection with on_change callback
        model_option = st.selectbox(
            'We are using CoralScope model', 
            ('Model_B'),
            key='model_option',
            on_change=on_model_change  # Call on_model_change when the selection changes
        )

        if  st.button('Segment Image'):
            st.session_state['segment'] = True
            # if st.session_state.get('segment'):
            masks = load_model_and_segment(coral_image, model_option)
            # print(type(masks),masks)
            list_of_images, titles = process_images(image=coral_image, masks=masks)
            
            # Create the plot and store it in session state
            create_plot(coral_image, masks)

            # Save the segmented images and titles in session state
            st.session_state['segmented_images'] = list_of_images
            st.session_state['titles'] = titles

            # Initial selection
            st.session_state['selected_index'] = 0

        if st.session_state.get('segment'):
            if 'plot_image' in st.session_state:
                # Display the stored plot
                st.image(st.session_state['plot_image'], caption='Original Image with Annotations')

            if 'segmented_images' in st.session_state:
                # Display all segmented images in a grid with 2 columns
                cols = st.columns(2)
                for i, img in enumerate(st.session_state['segmented_images']):
                    col = cols[i % 2]
                    with col:
                        st.image(img, caption=f'Image {i}', use_column_width=True)


                # User selects multiple segmented images
                selected_indices = st.multiselect('Select segmented images', 
                                                  range(len(st.session_state['segmented_images'])),
                                                  key='selected_image_indices')
                st.write("You selected:", selected_indices)

                if selected_indices:
                    if st.button('Generate Stacked Image'):
                        stacked_images = [st.session_state['segmented_images'][idx] for idx in selected_indices]
                        stacked_image = stack_images(stacked_images, direction="horizontal")
                        st.image(stacked_image, caption='Stacked Image', use_column_width=True)
                        st.session_state['stacked_image'] = stacked_image
                        
                # Button to trigger the histogram plot
                if st.button('Analyze colors in the selected image'):
                    st.session_state['plot_histogram'] = True

                if st.session_state.get('plot_histogram'):
                    if 'stacked_image' in st.session_state:
                        multiple_mask_output(st.session_state['stacked_image'])
                        OcrAnalysis.plot_custom_colorchart(color_chart)
                        st.session_state['plot_histogram'] = False
                    else:
                        st.write("Please generate a stacked image first.")






    # is_color_chart_in_session_state()
    # if st.button("Start Segmentation"):
    #     st.title("Perspective correction of the color chart")
    #     st.markdown("In this page, you can correct the perspective of the color chart.")
    #     # st.markdown("You can also add padding to the image to avoid cropping.")
    #     st.markdown("After adjusting the perspective of the image, you can save the image to memory.")
    #     st.markdown("You can also go back to the previous page to separate the color chart segments.")

    #     is_color_chart_in_session_state()
    #     image = st.session_state["chart_img"]

    #     pts = select_perspective_points(image.shape)
    #     rotated_image = perspective_correction(image, pts)
    #     st.image(rotated_image, caption="Warp Image", use_column_width=True)



    #     if st.button("Save Images to memory"):
    #         st.session_state["chart_img"] = rotated_image
    #     switch_to_color()
    # else:
    #     st.write("Please go to page 1 to upload the image and select the color chart")
    #     switch_to_cropping()

# Streamlit app execution
if __name__ == '__main__':
    main()