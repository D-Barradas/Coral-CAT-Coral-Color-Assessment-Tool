import streamlit as st
from PIL import Image
import io


import sys
sys.path.append('../')
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

from streamlit_extras.switch_page_button import switch_page


st.set_page_config(page_title="Mapping the custom color chart", page_icon="📊")

st.markdown("# Mapping the custom color chart")
st.sidebar.header("")

st.write(
    """This part is used to segment and create a mapped image"""
)

with open("load_functions.py") as f:
    exec(f.read())

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


def display_histogram(selected_index):
    """Function to display the histogram."""
    title= f'Image {selected_index}'
    top_RGB_colors = get_colors(st.session_state['segmented_images'][selected_index], 6, True)
    color_keys_selected, color_selected_distance, lower_y_limit, higher_y_limit, hex_colors_map = calculate_distances_to_colors(image=st.session_state['segmented_images'][selected_index])
    plot_compare(st.session_state['segmented_images'][selected_index], color_keys_selected, color_selected_distance, lower_y_limit, higher_y_limit, hex_colors_map, title)
    plot_compare_mapped_image(st.session_state['segmented_images'][selected_index], st.session_state['custom_color_chart'])

    # Define a function to be called when the model selection changes
def on_model_change():
    # Update the session state to indicate that the model has changed
    st.session_state.model_changed = True
    st.session_state['segment'] = False

# Initialize the session state
if 'model_changed' not in st.session_state:
    st.session_state['model_changed'] = False
    st.session_state['segment'] = False


def switch_to_cropping():
    """Must be one of ['streamlit starting page', 'cropping color chart', 'separate color chart', 'analysis and mapping', 'rotation of the color chart']"""

    want_to_contribute = st.button("Upload the image?")
    if want_to_contribute:
        switch_page("cropping color chart")




def main():
    if "coral_img" not in st.session_state or "custom_color_chart" not in st.session_state:
        st.write("Please go to page 1 to upload the image and page 2 to create the custom color chart")
        switch_to_cropping()

    else:
        st.write("Coral image and custom color chart are already in session state")

        coral_image = st.session_state["coral_img"]

        color_chart = st.session_state["custom_color_chart"]
        # Display the image
        st.image(coral_image, caption='Coral Image', use_column_width=True)


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
                # User selects from the segmented images
                selected_index = st.selectbox('Select a segmented image', 
                                            range(len(st.session_state['segmented_images'])),
                                            key='selected_image_index')
            
                # Display the selected segmented image
                st.image(st.session_state['segmented_images'][selected_index], 
                        caption=f'Image {selected_index}',
                        use_column_width=True)

                # Button to trigger the histogram plot
                if st.button('Analyze colors in the selected image'):
                    st.session_state['plot_histogram'] = True
            
                # Check if the histogram should be displayed
                if st.session_state.get('plot_histogram'):
                    display_histogram(selected_index)
                    
                    OcrAnalysis.plot_custom_colorchart(color_chart)

                    st.session_state['plot_histogram'] = False  # Reset the flag
            


# Streamlit app execution
if __name__ == '__main__':
    main()
