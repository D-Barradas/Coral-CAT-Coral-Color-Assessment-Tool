import streamlit as st
import matplotlib.pyplot as plt
from streamlit_extras.switch_page_button import switch_page
import sys ,os
from io import BytesIO
from zipfile import ZipFile
import pandas as pd
from pathlib import Path
import json
import numpy as np
# sys.path.append('../')
# from pathlib import Path

# # Add root directory to Python path
# ROOT_DIR = Path(__file__).parent.parent
# sys.path.append(str(ROOT_DIR))

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import time 

from load_functions import *
# with open("load_functions.py") as f:
#     exec(f.read())


def _mask_to_rle(mask):
    """Run-length encode a boolean mask for lightweight JSON storage."""

    flat = mask.astype(np.uint8).ravel(order="C")
    counts = []
    last = 0
    run = 0
    for v in flat:
        if v == last:
            run += 1
        else:
            counts.append(run)
            run = 1
            last = v
    counts.append(run)
    return counts, list(mask.shape)


def _normalize_meta_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def _masks_to_metadata(masks, image_name):
    records = []
    for idx, m in enumerate(masks):
        rec = {"image": image_name, "mask_id": idx}
        for key, value in m.items():
            if key == "segmentation":
                counts, shape = _mask_to_rle(value)
                rec["segmentation_rle"] = counts
                rec["segmentation_shape"] = shape
            else:
                rec[key] = _normalize_meta_value(value)
        records.append(rec)
    return records


def switch_to_color():
    """Must be one of ['streamlit starting page', 'upload image and define areas', 'build custom color chart', 'color analysis and mapping', 'rotation of the color chart']"""

    want_to_contribute = st.button("Go back to Separate the color chart segments?")
    if want_to_contribute:
        switch_page("build custom color chart")


def switch_to_manual():
    """Must be one of ['streamlit starting page', 'upload image and define areas', 'build custom color chart', 'color analysis and mapping', 'rotation of the color chart']"""

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
    sam_checkpoint = "checkpoints/vit_b_coralscop.pth"  # this is coralSCOPE
    model_type = "vit_b"

    if is_cuda_available():
        st.markdown("CUDA is available!")
        device = torch.device("cuda")  # reactivate the previous line for the app
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
        'Hex Color': hex_colors_map,
        'RGB Color': [tuple([int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)]) for hex_color in hex_colors_map]

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
        'Hex Color': hex_colors_map,
        'RGB Color': [tuple([int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)]) for hex_color in hex_colors_map]
    })

    # csv = color_distribution_data.to_csv(index=False).encode('utf-8')
    # csv = color_distribution_data.to_csv("Pie_color_chart.csv",index=False, encoding="utf-8")

    return fig, color_distribution_data


def get_colors_to_memory(image, number_of_colors):
    # Drop all black pixels from the image
    non_black_pixels = image[np.any(image != [0, 0, 0], axis=-1)]
    
    modified_image = non_black_pixels.reshape(non_black_pixels.shape[0], 3)
    # modified_image = image.reshape(image.shape[0]*image.shape[1], 3)

    clf = KMeans(n_clusters=number_of_colors, n_init='auto', random_state=73)
    labels = clf.fit_predict(modified_image)


    counts = Counter(labels)
    counts = dict(sorted(counts.items()))

    total_pixels = sum(counts.values())
    percentages = {k: (v / total_pixels) * 100 for k, v in counts.items()}

    center_colors = clf.cluster_centers_
    ordered_colors = [center_colors[i] for i in counts.keys()]
    hex_colors = [RGB2HEX(ordered_colors[i]) for i in counts.keys()]

    color_distribution_data = pd.DataFrame({
            'Color': list(counts.keys()),
            # 'Count': list(counts.values()),
            'Percentage': list(percentages.values()),
            'Hex': hex_colors,
            'RGB': [tuple(int(hex_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) for hex_color in hex_colors]
        })

        #     # Convert the DataFrame to a CSV string
        # csv = color_distribution_data.to_csv(index=False).encode('utf-8')

        # st.download_button(
        #     label="Download Color Distribution Data",
        #     data=csv,
        #     file_name="Pie_chart_color_distribution_data.csv",
        #     mime="text/csv",
        # )

    return color_distribution_data




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
            # cache SAM metadata in session_state for later use
            if "sam_metadata_by_image" not in st.session_state:
                st.session_state["sam_metadata_by_image"] = {}
            if "sam_metadata_records" not in st.session_state:
                st.session_state["sam_metadata_records"] = []
            sam_records = _masks_to_metadata(masks, uploaded_file.name)
            st.session_state["sam_metadata_by_image"][uploaded_file.name] = sam_records
            st.session_state["sam_metadata_records"].extend(sam_records)
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

                    # this section is for the color clustering distribution
                    # we will set the number of colors to 6 because is a good number of colors to detect on corals

                    csv_pie_chart = get_colors_to_memory(img, number_of_colors=6)
                    st.session_state[f"colors_detected_on_image_data_{name}_{idx}"] = csv_pie_chart 



                    #this section is for the euclidian distance
                    title = f"Image {idx} of {name}"
                    color_keys_selected, color_selected_distance, lower_y_limit, higher_y_limit, hex_colors_map = calculate_distances_to_colors(image=img, custom_color_chart=custom_color_chart)
                    fig_1, csv_1 = plot_compare_results_to_memory(img, color_keys_selected, color_selected_distance, lower_y_limit, higher_y_limit, hex_colors_map, title)
                    
                    st.session_state[f"euclidian_distance_{name}_{idx}"] = fig_1 
                    st.session_state[f"clustering_color_data_{name}_{idx}"] = csv_1


                    # This section is for the color mapping
                    # plot_compare_mapped_image_batch_mode(list_of_images[0],custom_color_chart,idx)
                    fig , csv = plot_compare_mapped_image_batch_mode_results_to_memory( img , custom_color_chart)
                    # save fig and csv into a dictionary that dictionary will be saved in the session state
                    st.session_state[f"mapped_image_{name}_{idx}"] = fig 
                    st.session_state[f"color_distribution_data_{name}_{idx}"] = csv

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
                    image_path_cluster = f"{key.replace('mapped_image', 'euclidian_distance')}.png"
                    image_cluster.savefig(image_path_cluster, format="png")
                    z.write(image_path_cluster)
                    os.remove(image_path_cluster)

                    # save the csv
                    csv = st.session_state.get(key.replace("mapped_image", "color_distribution_data"))
                    csv_path = f"{key.replace('mapped_image', 'color_distribution_data')}.csv"
                    csv.to_csv(csv_path, index=False)
                    z.write(csv_path)
                    os.remove(csv_path)
                    # z.writestr(csv_path, csv)
                    
                    # save the other csv
                    csv_cluster = st.session_state.get(key.replace("mapped_image", "clustering_color_data"))
                    csv_path_cluster = f"{key.replace('mapped_image', 'clustering_color_data')}.csv"
                    csv_cluster.to_csv(csv_path_cluster, index=False)
                    z.write(csv_path_cluster)
                    os.remove(csv_path_cluster)
                    # z.writestr(csv_path, csv_cluster)

                    # save the csv fro pie chart
                    csv_pie_chart = st.session_state.get(key.replace("mapped_image", "colors_detected_on_image_data"))
                    csv_path_pie_chart = f"{key.replace('mapped_image', 'colors_detected_on_image_data')}.csv"
                    csv_pie_chart.to_csv(csv_path_pie_chart, index=False)
                    z.write(csv_path_pie_chart)
                    os.remove(csv_path_pie_chart)
                    # z.writestr(csv_path, csv_pie_chart)

        # Download the zip file containing both images and CSVs
        st.download_button(
            label="Download Results zip file",
            data=results_zip.getvalue(),
            file_name="results.zip",
            mime="application/zip"
        )

    # Optional: compute real area (mm^2) for SAM masks using cached rulers
    st.markdown("---")
    st.header("Optional: compute real areas from rulers")
    compute_opt_in = st.checkbox("Compute area_mm2 using cached rulers", value=False)
    if compute_opt_in:
        sam_meta_path = Path("data/interim/benchmark/sam/sam_segments_metadata.csv")

        sam_source = "csv"
        if not sam_meta_path.exists():
            if "sam_metadata_records" in st.session_state and st.session_state["sam_metadata_records"]:
                st.write("Using SAM metadata from session_state cache.")
                sam_source = "session"
            else:
                st.write(f"SAM metadata not found: {sam_meta_path}")
                sam_source = "missing"
        else:
            st.write(f"Found SAM metadata: {sam_meta_path}")

        # allow optional upload of a rulers CSV to override cached rulers
        uploaded_rulers = st.file_uploader("(Optional) Upload rulers CSV to use instead", type=["csv"])

        if st.button("Compute area_mm2 now"):
            # Clean up any existing output file from previous runs
            cleanup_path = sam_meta_path.parent / 'sam_segments_with_area_mm2.csv'
            if cleanup_path.exists():
                try:
                    cleanup_path.unlink()
                except Exception:
                    pass
            
            try:
                if sam_source == "csv":
                    sam_df = pd.read_csv(sam_meta_path)
                elif sam_source == "session":
                    sam_df = pd.DataFrame(st.session_state.get("sam_metadata_records", []))
                else:
                    st.error("No SAM metadata available. Run segmentation or provide the metadata CSV.")
                    sam_df = None
            except Exception as e:
                st.error(f"Could not read SAM metadata: {e}")
                sam_df = None

            if sam_df is not None:
                if uploaded_rulers:
                    try:
                        rulers_df = pd.read_csv(uploaded_rulers)
                    except Exception as e:
                        st.error(f"Could not read uploaded rulers CSV: {e}")
                        rulers_df = pd.DataFrame()
                elif "ruler_records" in st.session_state and st.session_state["ruler_records"]:
                    rulers_df = pd.DataFrame(st.session_state["ruler_records"])
                else:
                    st.warning("No cached rulers found and none uploaded. Aborting computation.")
                    rulers_df = pd.DataFrame()

                # Hybrid smart matching: exact match + stem match + single ruler fallback
                if not rulers_df.empty and 'mm_per_px' in rulers_df.columns:
                    # Extract basenames (with extension) and stems (without extension)
                    rulers_df['image_basename'] = rulers_df['image'].fillna('').apply(lambda x: Path(str(x)).name.lower())
                    rulers_df['image_stem'] = rulers_df['image'].fillna('').apply(lambda x: Path(str(x)).stem.lower())
                    
                    sam_df['image_basename'] = sam_df['image'].fillna('').apply(lambda x: Path(str(x)).name.lower())
                    sam_df['image_stem'] = sam_df['image'].fillna('').apply(lambda x: Path(str(x)).stem.lower())
                    
                    # Build lookup dictionaries
                    rulers_by_basename = rulers_df.set_index('image_basename')['mm_per_px'].to_dict()
                    rulers_by_stem = rulers_df.set_index('image_stem')['mm_per_px'].to_dict()
                    
                    # Pass 1: Try exact basename match (with extension)
                    sam_df['mm_per_px'] = sam_df['image_basename'].map(rulers_by_basename)
                    
                    # Pass 2: For unmatched, try stem match (without extension)
                    unmatched_mask = sam_df['mm_per_px'].isna()
                    sam_df.loc[unmatched_mask, 'mm_per_px'] = sam_df.loc[unmatched_mask, 'image_stem'].map(rulers_by_stem)
                    
                    # Pass 3: If only 1 unique ruler exists, apply to remaining unmatched
                    unique_rulers = rulers_df['mm_per_px'].nunique()
                    unmatched_after_pass2 = sam_df['mm_per_px'].isna()
                    
                    if unique_rulers == 1 and unmatched_after_pass2.any():
                        default_mm_per_px = float(rulers_df['mm_per_px'].iloc[0])
                        num_applied = unmatched_after_pass2.sum() // len(sam_df[['image']].drop_duplicates())  # rough count of images
                        sam_df.loc[unmatched_after_pass2, 'mm_per_px'] = default_mm_per_px
                        st.info(f"ℹ️ Applied single ruler ({default_mm_per_px:.6f} mm/px) to unmatched images.")
                    
                    # Create diagnostic matching summary
                    unique_images = sam_df[['image']].drop_duplicates().sort_values('image')
                    match_status_list = []
                    
                    for img_name in unique_images['image'].values:
                        img_basename = sam_df[sam_df['image'] == img_name]['image_basename'].iloc[0]
                        img_stem = sam_df[sam_df['image'] == img_name]['image_stem'].iloc[0]
                        mm_val = sam_df[sam_df['image'] == img_name]['mm_per_px'].iloc[0]
                        
                        if img_basename in rulers_by_basename:
                            status = "✓ Exact (with ext)"
                        elif img_stem in rulers_by_stem:
                            status = "✓ Matched (no ext)"
                        elif pd.notna(mm_val) and unique_rulers == 1:
                            status = "✓ Default rule"
                        else:
                            status = "❌ No match"
                        
                        match_status_list.append({
                            'Image': img_name,
                            'mm_per_px': f"{mm_val:.6f}" if pd.notna(mm_val) else "—",
                            'Status': status
                        })
                    
                    match_summary_df = pd.DataFrame(match_status_list)
                    
                    # Display diagnostic table
                    st.subheader("📊 Ruler Matching Summary")
                    st.dataframe(match_summary_df, use_container_width=True, hide_index=True)
                    
                    # Check for any unmatched
                    failed_count = (match_summary_df['Status'] == '❌ No match').sum()
                    if failed_count > 0:
                        st.warning(f"⚠️ {failed_count} image(s) could not be matched to rulers. Download may have empty mm_per_px values.")
                    else:
                        st.success(f"✅ All {len(match_summary_df)} image(s) matched successfully!")
                    
                    # Continue with area computation
                    sam_df['area_px'] = pd.to_numeric(sam_df.get('area', sam_df.get('area_px', pd.Series())), errors='coerce')
                    sam_df['area_mm2'] = sam_df.apply(lambda r: r['area_px'] * (r['mm_per_px'] ** 2) if pd.notna(r.get('mm_per_px')) and pd.notna(r.get('area_px')) else pd.NA, axis=1)

                    # Select only required columns for compact output
                    compact_columns = ['image', 'mask_id', 'image_basename', 'mm_per_px', 'area_px', 'area_mm2']
                    available_cols = [col for col in compact_columns if col in sam_df.columns]
                    compact_df = sam_df[available_cols]
                    
                    # Create CSV in memory
                    csv_bytes = compact_df.to_csv(index=False).encode('utf-8')
                    st.success(f"✅ Computed area_mm2 for {len(compact_df)} mask segments.")
                    st.download_button("Download segments with area (CSV)", data=csv_bytes, file_name="sam_segments_with_area_mm2.csv", mime='text/csv')

                else:
                    st.warning('No valid rulers data found (missing mm_per_px). Save a ruler via the app or upload a valid CSV.')

        # allow downloading cached metadata as JSON
        if "sam_metadata_records" in st.session_state and st.session_state["sam_metadata_records"]:
            meta_json = json.dumps(st.session_state["sam_metadata_records"], separators=(",", ":"))
            st.download_button("Download SAM metadata (JSON)", data=meta_json, file_name="sam_segments_metadata_cache.json", mime="application/json")



            



# Streamlit app execution
if __name__ == '__main__':
    main()




