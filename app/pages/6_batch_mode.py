import streamlit as st
import matplotlib.pyplot as plt
from streamlit_extras.switch_page_button import switch_page
import sys ,os
from io import BytesIO
from zipfile import ZipFile
sys.path.append('../')
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import time 

with open("load_functions.py") as f:
    exec(f.read())


def switch_to_color():
    """Must be one of ['streamlit starting page', 'cropping color chart', 'separate color chart', 'analysis and mapping', 'rotation of the color chart']"""

    want_to_contribute = st.button("Go back to Separate the color chart segments?")
    if want_to_contribute:
        switch_page("separate color chart")


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
def load_model(model_option='Model_B'):
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
    # masks = mask_generator.generate(image)
    return mask_generator


def plot_compare_mapped_image_batch_mode_results_to_memory(img1_rgb, color_map_RGB):
    # check if the black color is in the color map if not add it
    if 'Black' not in color_map_RGB.keys():
        color_map_RGB['Black'] = tuple([0, 0, 0])

    mapped_image, color_map, color_to_pixels = map_color_to_pixels(image=img1_rgb, color_map_RGB=color_map_RGB)
    if 'Black' in color_map.keys():
        del color_map['Black']
    if 'Black' in color_to_pixels.keys():
        del color_to_pixels['Black']

    color_counts, reverse_dict = count_pixel_colors(image=mapped_image, color_map_RGB=color_map)
    lists = sorted(reverse_dict.items(), key=lambda kv: kv[1], reverse=True)

    color_name, percentage_color_name = [], []
    for c, p in lists:
        if p > 1:
            color_name.append(c)
            percentage_color_name.append(p)

    hex_colors_map = [RGB2HEX(color_map[key]) for key in color_name]

    # Create a subplot grid with adjusted row widths and column widths
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))

    # Add the original image, mapped image, and the bar chart to respective subplots
    axes[0].imshow(img1_rgb)
    axes[0].set_title("Original")
    axes[0].axis('off')

    axes[1].imshow(mapped_image)
    axes[1].set_title("Mapped Image")
    axes[1].axis('off')

    axes[2].bar(color_name, percentage_color_name, color=hex_colors_map)
    axes[2].set_title("Color Distribution")
    axes[2].set_xlabel("Color code in chart")
    axes[2].set_ylabel("Percentage of pixel on the image")

    plt.tight_layout()
    # close the plot
    plt.close()

    # Convert the color distribution data into a DataFrame
    color_distribution_data = pd.DataFrame({
        'Color Name': color_name,
        'Percentage': percentage_color_name,
        'Hex Color': hex_colors_map
    })

    # Convert the DataFrame to a CSV string
    # csv = color_distribution_data.to_csv(index=False).encode('utf-8')
    # csv = color_distribution_data.to_csv("color_distribution_data.csv",index=False, encoding = "utf-8")
    return fig, color_distribution_data


def plot_compare_results_to_memory(img1_rgb, color_keys_selected, color_selected_distance, lower_y_limit, higher_y_limit, hex_colors_map, title):
    # Convert black pixels to white in the image to show
    img1_rgb = convert_black_to_white(img1_rgb)

    # Create a subplot grid with 1 row and 2 columns
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Add the image to the first subplot
    axes[0].imshow(img1_rgb)
    axes[0].set_title(title)
    axes[0].axis('off')

    # Add the bar chart to the second subplot
    axes[1].bar(color_keys_selected, color_selected_distance, color=hex_colors_map)
    axes[1].set_title("Euclidean Distance from Top 5 Colors Detected")
    axes[1].set_xlabel("Color code in chart")
    axes[1].set_ylabel("Euclidean Distance")
    axes[1].set_ylim([lower_y_limit, higher_y_limit])

    plt.tight_layout()
    plt.close()

    # Create a csv file with the color distribution data
    color_distribution_data = pd.DataFrame({
        'Color Name': color_keys_selected,
        'Euclidean Distance': color_selected_distance,
        'Hex Color': hex_colors_map
    })

    # csv = color_distribution_data.to_csv(index=False).encode('utf-8')
    # csv = color_distribution_data.to_csv("Pie_color_chart.csv",index=False, encoding="utf-8")

    return fig, color_distribution_data




def main():
    st.title("Batch Mode")
    is_color_chart_in_session_state()
    # images = []
    # csvs = []
    uploaded_files = st.file_uploader("Choose the images ...", type=["bmp", "jpg", "jpeg", "png", "svg"], accept_multiple_files=True)
    for uploaded_file in uploaded_files:
        # bytes_data = uploaded_file.read()
        st.write("filename:", uploaded_file.name)

    if st.button("Start Segmentation"):
        mask_generator = load_model()
        progress_text = "Operation in progress. Please wait."
        my_bar = st.progress(0, text=progress_text)
        total_time = 0  # Initialize total time
        num_images = 0  # Initialize number of images processed

        for idx_f,uploaded_file in enumerate (uploaded_files)  :
            name = uploaded_file.name.split(".")[0]


            custom_color_chart = st.session_state["custom_color_chart"]
            # print (custom_color_chart.keys() ,"for loop")
            # if idx > 0 :
            #     #add black to the custom color chart
            #     custom_color_chart['Black'] =tuple([0,0,0])

            # for each image in the uploaded files we will apply the same process 
            # use get_image function to get the image
            # then use load_model_and_segment
            # then process_images 

            image = get_image(uploaded_file)
            masks = mask_generator.generate(image)
            # at this point we have the masks and the image crops 
            list_of_images, titles = process_images(image, masks)

            if len(list_of_images) > 1:
                st.write(f"Warning {len(list_of_images)} coral images detected on image:{name}")


            # if len(list_of_images) > 1: # we have to change this for the for look 
            with st.status(f"Processing images of {name} ...", expanded=True) as status:
                for idx , img in enumerate ( list_of_images) :
                    start_time = time.time()  # Record the start time 
                    # relocate the idx to the for loop here and add the name of the image
                    # relocate also the st.session_state[f"mapped_image_{idx}_{name}"] = fig
                    # relocate also the st.session_state[f"color_distribution_data_{idx}_{name}"] = csv
                    # we have to save the names of the images in a list to use it on the download button
                    # 
                    title = f"Image {idx} of {name}"
                    color_keys_selected, color_selected_distance, lower_y_limit, higher_y_limit, hex_colors_map = calculate_distances_to_colors(image=img)
                    fig_1, csv_1 = plot_compare_results_to_memory(img, color_keys_selected, color_selected_distance, lower_y_limit, higher_y_limit, hex_colors_map, title)
                    
                    st.session_state[f"euclidian_distance_{idx}_{name}"] = fig_1 
                    st.session_state[f"clustering_color_data_{idx}_{name}"] = csv_1


                    # now we will plot the images
                    # plot_compare_mapped_image_batch_mode(list_of_images[0],custom_color_chart,idx)
                    fig , csv = plot_compare_mapped_image_batch_mode_results_to_memory( img , custom_color_chart)
                    # save fig and csv into a dictionary that dictionary will be saved in the session state
                    st.session_state[f"mapped_image_{idx}_{name}"] = fig 
                    st.session_state[f"color_distribution_data_{idx}_{name}"] = csv

                    end_time = time.time()  # Record the end time
                    elapsed_time = end_time - start_time  # Calculate the elapsed time
                    total_time += elapsed_time  # Update total time
                    # num_images += 1  # Update number of images processed
                    st.write(f"Time spent processing image {idx} of {name}: {elapsed_time:.2f} seconds")

                status.update(label=f"Process complete for {name}!", state="complete", expanded=False)

            # show the progress bar here
            percent_complete = (idx_f + 1) / len(uploaded_files)
            my_bar.progress(percent_complete , text=progress_text)
        # Calculate and display total and average time
        # average_time = total_time / num_images if num_images > 0 else 0
        st.write(f"Total time spent processing: {total_time:.2f} seconds ({total_time / 60:.2f} minutes)")
        # st.write(f"Average time per image: {average_time:.2f} seconds ({average_time / 60:.2f} minutes)")

    # here there is a button to download the results
    if st.button("Process Results"):
        results_zip = BytesIO()
        with ZipFile(results_zip, 'w') as z:
            for key in st.session_state.keys():
    
                if "mapped_image" in key:
                    # st.write(f"{key}.png")
                    # st.write(f"{key.replace("mapped_image", "euclidian_distance")}.png")
                    # st.write(f"{key.replace("mapped_image", "color_distribution_data")}.csv")
                    # st.write(f"{key.replace("mapped_image", "clustering_color_data")}.csv")

                    image = st.session_state.get(key)
                    image_path = f"{key}.png"
                    image.savefig(image_path, format="png")
                    z.write(image_path)
                    os.remove(image_path)

                    # save the other plot
                    image_cluster = st.session_state.get(key.replace("mapped_image", "euclidian_distance"))
                    image_path_cluster = f"{key.replace("mapped_image", "euclidian_distance")}.png"
                    image_cluster.savefig(image_path_cluster, format="png")
                    z.write(image_path_cluster)
                    os.remove(image_path_cluster)

                    # save the csv
                    csv = st.session_state.get(key.replace("mapped_image", "color_distribution_data"))
                    csv_path = f"{key.replace("mapped_image", "color_distribution_data")}.csv"
                    csv.to_csv(csv_path, index=False)
                    z.write(csv_path)
                    os.remove(csv_path)
                    # z.writestr(csv_path, csv)
                    
                    # save the other csv
                    csv_cluster = st.session_state.get(key.replace("mapped_image", "clustering_color_data"))
                    csv_path_cluster = f"{key.replace("mapped_image", "clustering_color_data")}.csv"
                    csv_cluster.to_csv(csv_path_cluster, index=False)
                    z.write(csv_path_cluster)
                    os.remove(csv_path_cluster)
                    # z.writestr(csv_path, csv_cluster)

        # Download the zip file containing both images and CSVs
        st.download_button(
            label="Download Results zip file",
            data=results_zip.getvalue(),
            file_name="results.zip",
            mime="application/zip"
        )



            



# Streamlit app execution
if __name__ == '__main__':
    main()




