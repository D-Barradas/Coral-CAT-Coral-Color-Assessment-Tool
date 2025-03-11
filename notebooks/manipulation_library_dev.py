import numpy as np
import matplotlib.pyplot as plt
import cv2
import os 
from glob import glob 
from collections import Counter, defaultdict
from skimage.color import rgb2lab, deltaE_cie76
import pandas as pd
from sklearn.cluster import KMeans


def get_image(image_path):
    image = cv2.imread(image_path)
    # image = white_balance(img=image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image = preprocess_histograms( image=image)
    return image

## Extract and analyze colors
def RGB2HEX(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))

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

def background_to_black ( image, index ):
    # Apply the mask to the image
    masked_img = image.copy()
    # masked_pixels = masked_img[masks[index]['segmentation']==True]
    # masked_img[masks[index]['segmentation']==False] = (0, 0, 0)  # Set masked pixels to black
    masked_pixels = masked_img[index['segmentation']==True]
    masked_img[index['segmentation']==False] = (0, 0, 0)  # Set masked pixels to black
    return masked_img ,masked_pixels


# def closest_color(pixel, palette, palette_keys, color_map_RGB):
#     pixel_lab = rgb2lab(np.uint8(pixel))
#     distances = deltaE_cie76(pixel_lab, palette)
#     closest_index = np.argmin(distances)
#     return color_map_RGB[palette_keys[closest_index]]

# def minimal_distance(pixel, palette):
#     pixel_lab = rgb2lab(np.uint8(pixel))
#     distances = deltaE_cie76(pixel_lab, palette)
#     min_distance = np.min(distances)
#     return min_distance

# def label_minimal_distance(pixel, palette,palette_keys):
#     pixel_lab = rgb2lab(np.uint8(pixel))
#     distances = deltaE_cie76(pixel_lab, palette)
#     closest_index = np.argmin(distances)
#     return palette_keys[closest_index]


# def map_color_to_pixels(image,color_map_RGB):
#     # color_map_RGB = {'Black':(0,0,0),'White':(255,255,255),'B1': (247, 248, 232),'B2': (243, 244, 192),'B3': (234, 235, 137),'B4': (200, 206, 57),'B5': (148, 157, 56),
#     #                 'B6': (92, 116, 52),'C1': (247, 235, 232),'C2': (246, 201, 192),'C3': (240, 156, 136),'C4': (207, 90, 58),'C5': (155, 50, 32),'C6': (101, 27, 13),
#     #                 'D1': (246, 235, 224),'D2': (246, 219, 191),'D3': (239, 188, 135),'D4': (211, 147, 78),'D5': (151, 89, 36),'D6': (106, 58, 22),'E1': (247, 242, 227),
#     #                 'E2': (246, 232, 191),'E3': (240, 213, 136),'E4': (209, 174, 68),'E5': (155, 124, 45),'E6': (111, 85, 34)}
    
#     # Convert the colors in the color map to LAB space
#     palette = np.array([rgb2lab(np.uint8(np.asarray(color_map_RGB[key]))) for key in color_map_RGB.keys()])
#     palette_keys = list(color_map_RGB.keys())

#     # Function to apply to each pixel
#     func = lambda pixel: closest_color(pixel, palette, palette_keys, color_map_RGB)

#     # Apply the function to each pixel
#     mapped_img = np.apply_along_axis(func, -1, image)

#     # Function to apply to each pixel
#     func = lambda pixel: minimal_distance(pixel, palette)

#     # Apply the function to each pixel
#     mapped_dist = np.apply_along_axis(func, -1, image)

#     # Function to apply to each pixel
#     func = lambda pixel: label_minimal_distance(pixel, palette, palette_keys)

#     # Apply the function to each pixel
#     mapped_labels = np.apply_along_axis(func, -1, image)

#     # Create a dictionary where the keys are the colors in the mapped_img array
#     # and the values are lists of corresponding pixels in mapped_dist
#     mapped_dist_ravel = mapped_dist.ravel()
#     mapped_labels_ravel = mapped_labels.ravel()
#     # print (mapped_labels_ravel.shape , mapped_dist_ravel.shape )
#     assert mapped_labels_ravel.shape == mapped_dist_ravel.shape, "The two arrays must have the same shape."

#     color_to_pixels = defaultdict(list)
#     for idx,color in enumerate (mapped_labels_ravel):
#         color_to_pixels[color].append(mapped_dist_ravel[idx] )

#     return mapped_img ,color_map_RGB , color_to_pixels

def process_pixel(pixel, palette, palette_keys, color_map_RGB):
    pixel_lab = rgb2lab(np.uint8(np.asarray(pixel)))
    distances = deltaE_cie76(pixel_lab, palette)
    min_distance = np.min(distances)
    closest_index = np.argmin(distances)
    closest_color = color_map_RGB[palette_keys[closest_index]]
    label_min_distance = palette_keys[closest_index]
    return closest_color, min_distance, label_min_distance

def map_color_to_pixels(image, color_map_RGB):
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

def plot_compare_mapped_image_save(img1_rgb,filename,color_map_RGB):

    # get the mapped image 
    mapped_image , color_map , color_to_pixels = map_color_to_pixels(image=img1_rgb, color_map_RGB=color_map_RGB )
    del color_map['Black'] 
    del color_to_pixels['Black']
    # for color, pixels in color_to_pixels.items():
    #     print(f"Color: {color}")
    #     print(f"Max: {np.max(pixels)}")
    #     print(f"Min: {np.min(pixels)}")
    #     print(f"Std: {np.std(pixels)}")
    #     print(f"Mean: {np.mean(pixels)}")

    color_counts, reverse_dict = count_pixel_colors(image=mapped_image , color_map_RGB=color_map)

    # lists = sorted(reverse_dict.items()) # sorted by key, return a list of tuples
    lists = sorted(reverse_dict.items(), key=lambda kv: kv[1], reverse=True)

    # color_name, percentage_color_name = zip(*lists) # unpac the tupple
    color_name, percentage_color_name , max_val , min_val , mean_val , std_val  = [],[], [],[],[], []
    for c , p in lists:
        # print (c,p)
        if p > 1 :
            color_name.append(c)
            percentage_color_name.append(p)
            max_val.append ( np.max(color_to_pixels[c]) )
            min_val.append ( np.min(color_to_pixels[c]) ) 
            mean_val.append( np.mean(color_to_pixels[c]))
            std_val.append( np.std(color_to_pixels[c]))

    hex_colors_map = [RGB2HEX(color_map[key]) for key in color_name]

    results_df = pd.DataFrame({
    'color_name': color_name,
    'percentage_color_name': percentage_color_name,
    'hex_colors_map': hex_colors_map,
    'Mean_dist_val': mean_val,
    'Std_dist_val': std_val , 
    'Max_dist_val': max_val,
    'Min_dist_val': min_val
    })
    result_name = filename.replace(".jpg",".csv")
    results_df.to_csv(result_name)



    # Create a figure and subplots
    fig, (ax1, ax2, ax3 ) = plt.subplots(nrows=1,ncols=3,figsize=(15, 5))  # Adjust figsize as needed
    plt.title(label=filename.split("/")[-1].split(".")[0])
    # Display the images
    ax1.imshow(img1_rgb)
    ax1.set_title("Original")
    ax1.axis('off') 

    ax2.imshow(mapped_image)
    ax2.set_title("Mapped Image")
    ax2.axis('off') 


    # ax2.set_ylabel("Mapped Image")
    # ax2.set_xlabel("Color code in chart")

    ax3.bar(color_name, percentage_color_name, color = hex_colors_map , edgecolor='black' )
    ax3.yaxis.grid(True, linestyle='--', which='major',color='grey', alpha=.25)
    ax3.set_xlabel("Color code in chart")
    ax3.set_ylabel("Percentage of pixel on the image")
    plt.xticks(rotation=90)
    
    # plt.xlabel("Color code in chart")
    # plt.ylim(lower_y_limit,higher_y_limit)

    # Adjust spacing between subplots
    plt.tight_layout()
    #
    plt.savefig(fname=filename ,transparent=True ,format='jpg', dpi=300)
    # use close to dont show all images at once 
    # plt.close()

def get_colors(image, number_of_colors, show_chart):
    
    modified_image = image.reshape( image.shape[0]*image.shape[1],3  )
        
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
        box_height = 50
        # Clamp coordinates to image boundaries
        top_left_y = max(0, y - box_height)
        top_left_x = x
        bottom_right_y = y
        bottom_right_x = min(w, image.shape[1])  # Clamp right edge to image width

        cropped_image = image[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
        return cropped_image
    
    @staticmethod
    def plot_custom_colorchart(custom_rgb_chart):
        # Calculate the number of squares based on the dictionary length
        num_squares = len(custom_rgb_chart)

        # Define figure size and square width
        fig, ax = plt.subplots(figsize=(10, num_squares * 0.15))
        square_width = 0.8

        # Iterate over the dictionary and plot squares
        for i, (color_name, color_value) in enumerate(custom_rgb_chart.items()):


            # Normalize color values for plotting
            normalized_color = [c / 255 for c in color_value]
            # print ( normalized_color[0], len( normalized_color[0]))


            # Calculate x position based on square width and offset
            x_pos = i * square_width

            # Create and plot the square
            square = plt.Rectangle(
                xy=(x_pos, 0), width=square_width, height=1, color=normalized_color
            )
            ax.add_patch(square)

            # Add color name label above the square
            ax.text(
                x_pos + square_width / 2,
                1.15,
                color_name,
                ha="center",
                va="center",
                fontsize=10,
                weight="bold",rotation=90
            )

        # Set axis limits and labels
        ax.set_xlim([0, num_squares * square_width])
        ax.set_ylim([-0.2, 1.3])
        ax.set_xlabel("Color Name")
        ax.set_ylabel("Color Chart")

        # Remove unnecessary ticks and grid
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)

        # Show the plot
        plt.tight_layout()
        plt.show()

