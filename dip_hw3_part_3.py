

import cv2
from frequency_filtering.filtering import Filtering

def display_image(window_name, image):
    """A function to display image"""
    cv2.namedWindow(window_name)
    cv2.imshow(window_name, image)
    cv2.waitKey(0)

def main():
    """ The main funtion that parses input arguments, calls the approrpiate
     fitlering method and writes the output image"""

    image_name = "noisy_lenna"
    input_image = cv2.imread("Image_with_periodic_noise.png", 0)
    rows, cols = input_image.shape

    filter_obj = Filtering(input_image)
    output = filter_obj.filter()

    # Write output file
    output_dir = 'output/'

    output_image_name = output_dir+image_name+"_filtered_image.jpg"
    cv2.imwrite(output_image_name, output[0])
    output_image_name = output_dir + image_name+"_dft.jpg"
    cv2.imwrite(output_image_name, output[1])
    output_image_name = output_dir + image_name + "_dft_filtered.jpg"
    cv2.imwrite(output_image_name, output[2])


if __name__ == "__main__":
    main()







