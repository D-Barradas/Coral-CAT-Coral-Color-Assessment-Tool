import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
import cv2
import os 
from glob import glob 
from sklearn.cluster import KMeans
from collections import Counter , defaultdict
from skimage.color import rgb2lab, deltaE_cie76
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from io import StringIO


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) 

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)


def preprocess_histograms(image):
    #seperating colour channels
    B = image[:,:,0] #blue layer
    G = image[:,:,1] #green layer
    R = image[:,:,2] #red layer
    # equalize the histograms 
    b_equi = cv2.equalizeHist(B)
    g_equi = cv2.equalizeHist(G)
    r_equi = cv2.equalizeHist(R)
    equi_im = cv2.merge([b_equi,g_equi,r_equi])
    return equi_im

# Modify the get_image function to work with Streamlit's uploaded file
def get_image(uploaded_file):
    # Read the file into a byte stream
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    
    # Decode the file bytes to an image
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Convert color from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Apply any preprocessing like histogram preprocessing
    # image = preprocess_histograms(image=image)

    # Get the dimensions of the image
    height, width = image.shape[:2]

    # Calculate the aspect ratio
    aspect_ratio = width / height

    # Define the new width and height while maintaining the aspect ratio
    new_width = 1800
    new_height = int(new_width / aspect_ratio)

    # Resize the image
    image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # image = cv2.resize(image, (3600, 2400), interpolation=cv2.INTER_AREA)

    
    return image

def show_images_grid(images, titles=None, figsize=(20, 20)):
    """Displays a grid of images with optional titles."""

    num_images = len(images)
    rows = int(num_images / 2)
    cols = 2

    # Create a figure and subplots
    fig, axes = plt.subplots(rows, cols, figsize=figsize)

    # Flatten the subplots array for easier iteration
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i < num_images:
            img = images[i]
            ax.imshow(img)
            # ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Convert to RGB for Matplotlib
            ax.axis('off')  # Hide axes

            if titles:
                ax.set_title(titles[i])
        else:
            ax.axis('off')  # Hide unused subplots

    plt.tight_layout()
    plt.show()

def background_to_black ( image, index ,masks ):
    # Apply the mask to the image
    masked_img = image.copy()
    masked_pixels = masked_img[masks[index]['segmentation']==True]
    masked_img[masks[index]['segmentation']==False] = (0, 0, 0)  # Set masked pixels to black
    return masked_img ,masked_pixels

# create a function to convert all the black pixels to white 
def convert_black_to_white(image):
    image = np.array(image)
    image[np.where((image == [0,0,0]).all(axis=2))] = [255,255,255]
    return image

def map_white_pixels(source_image, target_image):
    """Maps white pixels from the source image to the target image.

    Args:
        source_image: The source image as a NumPy array.
        target_image: The target image as a NumPy array.

    Returns:
        A modified target image with the white pixels from the source mapped onto it.
    """

    # Ensure both images have the same dimensions
    if source_image.shape != target_image.shape:
        raise ValueError("Source and target images must have the same dimensions.")

    # Create a mask for white pixels in the source image
    source_white_mask = np.all(source_image == 255, axis=-1)

    # Map white pixels from source to target based on their positions
    target_image[source_white_mask] = source_image[source_white_mask]

    return target_image

def get_sorted_by_area(image, anns):

    # print(type(image),type(anns),"get_sorte_by_area")


    area_list=[]
    cropped_image_dic ={}
    mask_number = [] 
    mask_pixles_dic = {}
    for i in range(len(anns)):
        x, y, width, height = anns[i]['bbox']
        area = anns[i]["area"]
        image_b, masked_pixels = background_to_black(image=image, index=i, masks=anns)
        cropped_image = image_b[int(y):int(y+height), int(x):int(x+width)]

        area_list.append(area)
        cropped_image_dic[i] = cropped_image
        mask_pixles_dic[i] = masked_pixels
        mask_number.append(i)
    df = pd.DataFrame([area_list,mask_number])
    df = df.T
    df.columns = ['area','mask_number']
    df.sort_values(by='area', ascending=False, inplace=True)
    df.dropna(inplace=True)
    return df , cropped_image_dic , mask_pixles_dic




def RGB2HEX(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))


def get_colors(image, number_of_colors, show_chart):
    # Drop all black pixels from the image
    non_black_pixels = image[np.any(image != [0, 0, 0], axis=-1)]
    
    modified_image = non_black_pixels.reshape(non_black_pixels.shape[0], 3)
    # modified_image = image.reshape(image.shape[0]*image.shape[1], 3)

    clf = KMeans(n_clusters=number_of_colors, n_init='auto', random_state=73)
    labels = clf.fit_predict(modified_image)

    counts = Counter(labels)
    counts = dict(sorted(counts.items()))

    center_colors = clf.cluster_centers_
    ordered_colors = [center_colors[i] for i in counts.keys()]
    hex_colors = [RGB2HEX(ordered_colors[i]) for i in counts.keys()]
    rgb_colors = [ordered_colors[i] for i in counts.keys()]

    if show_chart:
        fig = px.pie(names=list(counts.keys()), values=list(counts.values()), 
                     color_discrete_sequence=hex_colors,
                     title="Color Distribution")

        fig.update_traces(textinfo='percent+label')
        st.plotly_chart(fig)

        # add a download button for the color distribution data
        color_distribution_data = pd.DataFrame({
            'Color': list(counts.keys()),
            'Count': list(counts.values()),
            'Hex': hex_colors
        })

            # Convert the DataFrame to a CSV string
        csv = color_distribution_data.to_csv(index=False).encode('utf-8')

        st.download_button(
            label="Download Color Distribution Data",
            data=csv,
            file_name="Pie_chart_color_distribution_data.csv",
            mime="text/csv",
        )

    return rgb_colors
    


def drop_black_from_top_colors(top_colors_list):
    min_values = []
    for i in range(len(top_colors_list)):
        curr_color = rgb2lab(np.uint8(np.asarray([[top_colors_list[i]]])))
        diff = deltaE_cie76((0, 0, 0), curr_color)
        # print (diff, type(diff))
        min_values.append(diff[0][0])
        lowest_value_index = np.argmin(min_values) 
    top_colors_list.pop(lowest_value_index)
    return top_colors_list


def match_image_by_color(image, color, threshold = 60, number_of_colors = 10): 
    
    image_colors = get_colors(image, number_of_colors, False)
    # discard black
    image_colors = drop_black_from_top_colors(image_colors)
    selected_color = rgb2lab(np.uint8(np.asarray([[color]])))

    diff_list =[]
    for i in range(len(image_colors)):
        curr_color = rgb2lab(np.uint8(np.asarray([[image_colors[i]]])))
        diff = deltaE_cie76(selected_color, curr_color)
        # print(diff[0][0])
        diff_list.append(diff[0][0])
    diff_avg = np.mean(diff_list)
    if diff_avg < threshold:
        return diff_avg
    else:
        # a euclidian difference of 1000 should be noticible 
        return 1000
    
def calculate_distances_to_colors(image):
    # color chart but in RGB 
    color_map_RGB = {
    'B1': (247, 248, 232),
    'B2': (243, 244, 192),
    'B3': (234, 235, 137),
    'B4': (200, 206, 57),
    'B5': (148, 157, 56),
    'B6': (92, 116, 52),
    'C1': (247, 235, 232),
    'C2': (246, 201, 192),
    'C3': (240, 156, 136),
    'C4': (207, 90, 58),
    'C5': (155, 50, 32),
    'C6': (101, 27, 13),
    'D1': (246, 235, 224),
    'D2': (246, 219, 191),
    'D3': (239, 188, 135),
    'D4': (211, 147, 78),
    'D5': (151, 89, 36),
    'D6': (106, 58, 22),
    'E1': (247, 242, 227),
    'E2': (246, 232, 191),
    'E3': (240, 213, 136),
    'E4': (209, 174, 68),
    'E5': (155, 124, 45),
    'E6': (111, 85, 34)
    }
    
    # get the distance 
    final_distances = {}
    for key in color_map_RGB.keys():
        max_val = match_image_by_color( image=image, color=color_map_RGB[key], number_of_colors=6)
        if max_val != 0 :
            final_distances[key]=max_val
    df_final = pd.DataFrame.from_dict(final_distances,orient='index',columns=["Distance"])
    df_final.sort_values(by="Distance",ascending=True,inplace=True)
    color_keys_selected = df_final.head(n=5).index.to_list()
    color_selected_distance = df_final["Distance"].head(n=5).to_list()
    lower_y_limit = color_selected_distance[0] - 0.5
    higher_y_limit = color_selected_distance[-1] + 0.5
    hex_colors_map = [RGB2HEX(color_map_RGB[key]) for key in color_keys_selected]
    return color_keys_selected, color_selected_distance, lower_y_limit, higher_y_limit,hex_colors_map


def plot_compare(img1_rgb, color_keys_selected, color_selected_distance, lower_y_limit, higher_y_limit, hex_colors_map,title):
    # Create a subplot grid with 1 row and 2 columns
    fig = make_subplots(rows=1, cols=2, subplot_titles=(title, "Euclidean Distance from Top 5 Colors Detected"))
    
    # Convert black pixels to white in the image to show
    img1_rgb = convert_black_to_white(img1_rgb)

    # Add the image to the first subplot
    fig.add_trace(go.Image(z=img1_rgb), row=1, col=1)

    # Add the bar chart to the second subplot
    fig.add_trace(go.Bar(x=color_keys_selected, y=color_selected_distance, marker_color=hex_colors_map), row=1, col=2)

    # Update y-axis range for the bar chart
    fig.update_yaxes(range=[lower_y_limit, higher_y_limit], row=1, col=2)

    # Update layout and axis properties
    fig.update_layout(showlegend=False)
    fig.update_xaxes(title_text="Color code in chart", row=1, col=2)
    fig.update_yaxes(title_text="Euclidean Distance", row=1, col=2)
    
    # Adjust layout for a better fit
    fig.update_layout(height=600, width=1100)

    # Show the plot
    st.plotly_chart(fig)

    # Create a csv file with the color distribution data
    color_distribution_data = pd.DataFrame({
        'Color Name': color_keys_selected,
        'Euclidean Distance': color_selected_distance,
        'Hex Color': hex_colors_map
    })

    # Convert the DataFrame to a CSV string
    csv = color_distribution_data.to_csv(index=False).encode('utf-8')
 # Create a download button and offer the CSV string for download
    st.download_button(
        label="Download Color Distribution Data",
        data= csv,
        file_name="Euclidean_Distance_from_Top_5_Colors_Detected.csv",
        mime="text/csv",
    )
    #fig.show()


def closest_color(pixel, palette, palette_keys, color_map_RGB):
    pixel_lab = rgb2lab(np.uint8(pixel))
    distances = deltaE_cie76(pixel_lab, palette)
    closest_index = np.argmin(distances)
    return color_map_RGB[palette_keys[closest_index]]


# def map_color_to_pixels(image):
#    """This is the original code , working but slow"""
#    # in this color map I added black and white to conserve the black 
#    color_map_RGB = {'Black':(0,0,0),'White':(255,255,255),'B1': (247, 248, 232),'B2': (243, 244, 192),'B3': (234, 235, 137),'B4': (200, 206, 57),'B5': (148, 157, 56),
#                     'B6': (92, 116, 52),'C1': (247, 235, 232),'C2': (246, 201, 192),'C3': (240, 156, 136),'C4': (207, 90, 58),'C5': (155, 50, 32),'C6': (101, 27, 13),
#                     'D1': (246, 235, 224),'D2': (246, 219, 191),'D3': (239, 188, 135),'D4': (211, 147, 78),'D5': (151, 89, 36),'D6': (106, 58, 22),'E1': (247, 242, 227),
#                     'E2': (246, 232, 191),'E3': (240, 213, 136),'E4': (209, 174, 68),'E5': (155, 124, 45),'E6': (111, 85, 34)}
#    palette = [ rgb2lab(np.uint8 ( np.asarray ( color_map_RGB[x] ))) for x in color_map_RGB.keys() ]
#    palette_keys = [ x for x in color_map_RGB.keys() ]
#    mapped_img = np.zeros_like(image)
#    for i in range(image.shape[0]):
#      for j in range(image.shape[1]):
#        pixel = rgb2lab(np.uint8 ( np.asarray ( image[i, j] ) ) )
#        distances = deltaE_cie76 ( lab1=pixel , lab2=palette )
#        closest_index = np.argmin(distances)  # Find closest color index
# # assing the color to the pixel in the mapped image
#        mapped_img[i, j] = color_map_RGB[palette_keys[closest_index]]

#    return mapped_img , color_map_RGB


# def map_color_to_pixels(image):
#     """ This version has a hardcoded color chart or idial chart as we called it, however this funtion add the use if apply_along_axis to make it faster"""
#     color_map_RGB = {'Black':(0,0,0),'White':(255,255,255),'B1': (247, 248, 232),'B2': (243, 244, 192),'B3': (234, 235, 137),'B4': (200, 206, 57),'B5': (148, 157, 56),
#                     'B6': (92, 116, 52),'C1': (247, 235, 232),'C2': (246, 201, 192),'C3': (240, 156, 136),'C4': (207, 90, 58),'C5': (155, 50, 32),'C6': (101, 27, 13),
#                     'D1': (246, 235, 224),'D2': (246, 219, 191),'D3': (239, 188, 135),'D4': (211, 147, 78),'D5': (151, 89, 36),'D6': (106, 58, 22),'E1': (247, 242, 227),
#                     'E2': (246, 232, 191),'E3': (240, 213, 136),'E4': (209, 174, 68),'E5': (155, 124, 45),'E6': (111, 85, 34)}
    
#     # Convert the colors in the color map to LAB space
#     palette = np.array([rgb2lab(np.uint8(np.asarray(color_map_RGB[key]))) for key in color_map_RGB.keys()])
#     palette_keys = list(color_map_RGB.keys())

#     # Function to apply to each pixel
#     func = lambda pixel: closest_color(pixel, palette, palette_keys, color_map_RGB)

#     # Apply the function to each pixel
#     mapped_img = np.apply_along_axis(func, -1, image)

#     return mapped_img ,color_map_RGB

def process_images(image, masks):
    """Process the images"""
    # print (type(image),type(masks), "process_images")

    image_dataframe, cropped_image_list , mask_pixels_dict = get_sorted_by_area( image=image , anns=masks )
    top_six_img_by_area = image_dataframe['mask_number'].head(n=10).to_list()
    list_of_images = [ cropped_image_list [idx ] for idx in top_six_img_by_area  ]
    titles = ['Image 1', 'Image 2', 'Image 3', 'Image 4', 'Image 5', 'Image 6','Image 7','Image 8','Image 9','Image 10']
    return list_of_images , titles


def process_pixel(pixel, palette, palette_keys, color_map_RGB):
    pixel_lab = rgb2lab(np.uint8(np.asarray(pixel)))
    distances = deltaE_cie76(pixel_lab, palette)
    min_distance = np.min(distances)
    closest_index = np.argmin(distances)
    closest_color = color_map_RGB[palette_keys[closest_index]]
    label_min_distance = palette_keys[closest_index]
    return closest_color, min_distance, label_min_distance


def map_color_to_pixels(image, color_map_RGB):
    """ This version uses a costume color chat to do the mapping"""
    palette = np.array([rgb2lab(np.uint8(np.asarray(color_map_RGB[key]))) for key in color_map_RGB.keys()])
    palette_keys = list(color_map_RGB.keys())

    # Apply the function to each pixel for each operation
    func_closest_color = lambda pixel: process_pixel(pixel, palette, palette_keys, color_map_RGB)[0]
    mapped_img = np.apply_along_axis(func_closest_color, -1, image)

    func_min_distance = lambda pixel: process_pixel(pixel, palette, palette_keys, color_map_RGB)[1]
    mapped_dist = np.apply_along_axis(func_min_distance, -1, image)

    func_label_min_distance = lambda pixel: process_pixel(pixel, palette, palette_keys, color_map_RGB)[2]
    mapped_labels = np.apply_along_axis(func_label_min_distance, -1, image)

    color_to_pixels = defaultdict(list)
    for idx, color in enumerate(mapped_labels.ravel()):
        color_to_pixels[color].append(mapped_dist.ravel()[idx])

    return mapped_img, color_map_RGB, color_to_pixels



def count_pixel_colors(image, color_map_RGB):
  """
  Counts the number of pixels of each color in an image.

  Args:
    image: A NumPy array representing the image.
    color_map_RGB: A dictionary mapping color names to RGB tuples.

  Returns:
    A dictionary mapping color names to the number of pixels of that color in the image.
  """
  # Flatten the image into a 1D array
  # image_flat = image.flatten()
  # return image_flat
  reverse_dict = { value : key for key , value in color_map_RGB.items() }  


  # iterate over the image pixels
  all_pixels_list =[]
  for i in range(image.shape[0]):
      for j in range(image.shape[1]):
        pixel = image[i, j]  
        # discard black 
        # if reverse_dict[str(pixel)] != 'Black':
        all_pixels_list.append(pixel)

  # # Count the occurrences of each pixel value
  pixel_counts = Counter(tuple(pixel_1) for pixel_1 in all_pixels_list)
  # delete the black key from the dictionary 
  del pixel_counts[(0,0,0)] 

  # pass the values to a list 
  total_pixels = [ item for key , item in pixel_counts.items() if key != (0,0,0)]
  # sum all the values 
  total_pixels = np.sum(total_pixels)
  # # Count the number of pixels of each color in the color map
  color_counts = {color_name: pixel_counts.get(color_rgb, 0)/total_pixels * 100 for color_rgb,color_name in reverse_dict.items()}

  return pixel_counts, color_counts 



# def plot_compare_mapped_image(img1_rgb):
    # # Assuming map_color_to_pixels and count_pixel_colors are defined elsewhere
    # mapped_image, color_map = map_color_to_pixels(image=img1_rgb)
    # del color_map['Black']



    # get the mapped image 
def plot_compare_mapped_image(img1_rgb,color_map_RGB):
    if 'Black' not in color_map_RGB.keys():
        color_map_RGB['Black'] = tuple([0, 0, 0])

    mapped_image , color_map , color_to_pixels = map_color_to_pixels(image=img1_rgb, color_map_RGB=color_map_RGB )
    del color_map['Black'] 
    del color_to_pixels['Black']


    color_counts, reverse_dict = count_pixel_colors(image=mapped_image , color_map_RGB=color_map)
    lists = sorted(reverse_dict.items(), key=lambda kv: kv[1], reverse=True)

    color_name, percentage_color_name = [], []
    for c, p in lists:
        if p > 1:
            color_name.append(c)
            percentage_color_name.append(p)

    hex_colors_map = [RGB2HEX(color_map[key]) for key in color_name]

    # apply convert_black_to_white
    img1_rgb = convert_black_to_white(img1_rgb)
    mapped_image = convert_black_to_white(mapped_image)
    # img1_rgb = map_white_pixels(source_image=mapped_image,target_image=img1_rgb)


    # Create a subplot grid with adjusted row widths and column widths
    fig = make_subplots(rows=1, cols=3, 
                        column_widths=[0.35, 0.35, 0.3],  # Adjust column widths
                        subplot_titles=("Original", "Mapped Image", "Color Distribution"))

    # Add the original image, mapped image, and the bar chart to respective subplots
    fig.add_trace(go.Image(z=img1_rgb), row=1, col=1)
    fig.add_trace(go.Image(z=mapped_image), row=1, col=2)
    fig.add_trace(go.Bar(x=color_name, y=percentage_color_name, marker_color=hex_colors_map), row=1, col=3)

    # Update layout and axis properties
    fig.update_layout(showlegend=False, height=600, width=1500)  # Adjust the total figure size
    fig.update_xaxes(title_text="Color code in chart", row=1, col=3)
    fig.update_yaxes(title_text="Percentage of pixel on the image", row=1, col=3)

    # Convert the color distribution data into a DataFrame
    color_distribution_data = pd.DataFrame({
        'Color Name': color_name,
        'Percentage': percentage_color_name,
        'Hex Color': hex_colors_map
    })
    
    # Convert the DataFrame to a CSV string
    csv = color_distribution_data.to_csv(index=False).encode('utf-8')
    
    
    # Display the plot in Streamlit
    st.plotly_chart(fig)

    # Create a download button and offer the CSV string for download
    st.download_button(
        label="Download Color Distribution Data",
        data=csv,
        file_name="color_distribution_data.csv",
        mime="text/csv",
    )

    

class OcrAnalysis:
    """Performs analysis on OCR (Optical Character Recognition) results.

    Attributes:
        None
    """

    def __init__(self):
        """Initializes the OcrAnalysis class."""
        pass

    @staticmethod
    def get_bounding_boxes(results):
        """Extracts bounding boxes and text from OCR results.

        Args:
            results: An iterable of tuples containing individual OCR results,
                each tuple having the format (bbox, text, prob) where:
                    - bbox: A list/tuple of coordinates representing the bounding box.
                    - text: The recognized text within the bounding box.
                    - prob: The confidence probability score (optional).

        Returns:
            A tuple of two lists:
                - The first list contains bounding boxes as NumPy arrays.
                - The second list contains the corresponding recognized text.
        """

        bboxes, text_list = [], []
        for bbox, text, _ in results:
            # Extract and convert coordinates to integers
            top_left, top_right, bottom_right, bottom_left = bbox
            box = np.array([int(coord) for coord in [top_left[0], top_left[1], bottom_right[0], bottom_right[1]]])
            bboxes.append(box)
            text_list.append(text)

        # sort the bboxes by the x coordinate in ascending order
        bboxes = [ 
            bbox for _, bbox in sorted(
                zip([box[0] for box in bboxes], bboxes)
                )
                ]

        text_list = [  
            text for _, text in sorted(
                zip([box[0] for box in bboxes], text_list)
                    )  
                    ]



        return bboxes, text_list

    @staticmethod
    def get_pixels_above_bbox(bbox, image):
        """Extracts the region above the given bounding box from an image.

        Args:
            bbox: A list/tuple representing the bounding box as [x, y, width, height].
            image: The NumPy array representing the image.

        Returns:
            A NumPy array containing the cropped image region.
        """

        x, y, w, h = bbox
        box_height = 100
        # Clamp coordinates to image boundaries
        top_left_y = max(0, y - box_height)
        top_left_x = x
        bottom_right_y = y
        bottom_right_x = min(w, image.shape[1])  # Clamp right edge to image width

        cropped_image = image[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
        return cropped_image
    
    @staticmethod
    def plot_custom_colorchart(custom_rgb_chart):

        # Prepare data for Plotly
        data = []
        for color_name, color_value in custom_rgb_chart.items():
            normalized_color = f'rgb({color_value[0]}, {color_value[1]}, {color_value[2]})'
            data.append({'Color Name': color_name, 'Color Value': normalized_color})

        # Create DataFrame
        df = pd.DataFrame(data)

        # Create Plotly Express figure
        fig = px.bar(df, x='Color Name', y=[1]*len(df), color='Color Value', 
                    color_discrete_map='identity', text='Color Name')

        # Update layout for better visualization
        fig.update_layout(
            showlegend=False,
            xaxis_title='Color Name',
            yaxis_title='Color Chart',
            yaxis=dict(showticklabels=False),
            xaxis=dict(tickangle=-90),
            height=400
        )
        # Display the plot in Streamlit
        st.plotly_chart(fig)


          


def crop_my_image(image,boxes,tag):
    ### this looks like it can include the mayority of the section we want to analyze
    # print(type(image))
    x, y, width, height =  boxes[tag]
    x, y, width, height = int(x),int(y),int(width), int(height)
    cropped_image = image[y:height,x:width]

    return cropped_image

def get_colors_df(image, number_of_colors, show_chart):

        # Drop all black pixels from the image
    non_black_pixels = image[np.any(image != [0, 0, 0], axis=-1)]
    
    modified_image = non_black_pixels.reshape(non_black_pixels.shape[0], 3)
    
    # modified_image = image.reshape( image.shape[0]*image.shape[1],3  )
        
    clf = KMeans(n_clusters = number_of_colors, n_init='auto', random_state=73)
    labels = clf.fit_predict(modified_image)
        
    counts = Counter(labels)
    # sort to ensure correct color percentage
    counts = dict(sorted(counts.items()))
    # print (counts)
    center_colors = clf.cluster_centers_

    # We get ordered colors by iterating through the keys
    ordered_colors = [center_colors[i] for i in counts.keys()]
    hex_colors = [RGB2HEX(ordered_colors[i]) for i in counts.keys()]
    rgb_colors = [ordered_colors[i] for i in counts.keys()]

    df_colors = pd.DataFrame({ #"ordered_colors":ordered_colors,
                "hex_colors":hex_colors,
                "counts_value":counts.values(),
                "rgb_colors":rgb_colors})
    # df_colors = df_colors[~df_colors['hex_colors'].str.contains("#0000")]
    df_colors['hex_colors'] = df_colors['hex_colors'].astype(str)

    df_colors['is_dark'] = df_colors['hex_colors'].apply(is_dark_color)
    df_colors = df_colors[df_colors['is_dark'] == False]
        # print (df_colors.info())


    if (show_chart):
        plt.figure(figsize = (8, 6))
        plt.pie(df_colors["counts_value"], labels= df_colors["rgb_colors"], colors=df_colors["hex_colors"])
            # plt.pie(counts.values(), labels = rgb_colors, colors = hex_colors)
        
    return df_colors.drop("counts_value",axis=1)

def is_dark_color(hex_code):
    """
    Determines whether a given hex color code represents a dark color.

    Args:
        hex_code (str): The hex color code (e.g., '#FF0000').

    Returns:
        bool: True if the color is considered dark, False otherwise.
    """

    r, g, b = tuple(int(hex_code.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
    # Calculate a weighted average of the RGB components, considering human eye sensitivity
    luminosity = (0.299 * r + 0.587 * g + 0.114 * b) / 255

    # Threshold based on luminance and desired darkness level
    return luminosity < 0.05  # Adjust this threshold as needed

## new definitions 

def add_padding_to_images(images, target_shape):
    """
    Adds padding to a list of images to match the target shape.

    Args:
        images (list): List of numpy.ndarray images to pad.
        target_shape (tuple): Target shape (height, width) to pad images to.

    Returns:
        padded_images: List of padded images.
    """
    padded_images = []
    target_height, target_width = target_shape

    for img in images:
        height, width, _ = img.shape
        top_pad = (target_height - height) // 2
        bottom_pad = target_height - height - top_pad
        left_pad = (target_width - width) // 2
        right_pad = target_width - width - left_pad

        padded_img = cv2.copyMakeBorder(
            img, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )
        padded_images.append(padded_img)

    return padded_images

def stack_images(images, direction='horizontal'):
    """
    Stacks a list of numpy.ndarray images either horizontally or vertically.

    Args:
        images (list): List of numpy.ndarray images to stack.
        direction (str): Direction to stack images ('horizontal' or 'vertical').

    Returns:
        stacked_image: The stacked image as a numpy.ndarray.
    """
    if direction not in ['horizontal', 'vertical']:
        raise ValueError("Direction must be 'horizontal' or 'vertical'")

    # Determine the target shape for padding
    max_height = max(img.shape[0] for img in images)
    max_width = max(img.shape[1] for img in images)
    target_shape = (max_height, max_width)

    # Add padding to images to match the target shape
    padded_images = add_padding_to_images(images, target_shape)

    if direction == 'horizontal':
        stacked_image = np.hstack(padded_images)
    else:
        stacked_image = np.vstack(padded_images)

    return stacked_image


