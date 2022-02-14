# For this part of the assignment, You can use inbuilt functions to compute the fourier transform
# You are welcome to use fft that are available in numpy and opencv

import numpy as np


class Filtering:

    def __init__(self, image):
        """initializes the variables frequency filtering on an input image
        takes as input:
        image: the input image
        """
        self.image = image
        self.mask = self.get_mask

    def get_mask(self, shape):
        """Computes a user-defined mask
        takes as input:
        shape: the shape of the mask to be generated
        rtype: a 2d numpy array with size of shape
        """
        mask = np.ones(shape)
        mask[225:245, 265:285] = 0
        mask[265:285,220:240]=0
        mask[225:235, 55:68] = 0
        mask[450:460, 275:295] = 0
        mask[272:289, 448:463] = 0
        mask[55:68, 220:240] = 0
        return mask

    def post_process_image(self, image):
        """Post processing to display DFTs and IDFTs
        takes as input:
        image: the image obtained from the inverse fourier transform
        return an image with full contrast stretch
        -----------------------------------------------------
        You can perform post processing as needed. For example,
        1. You can perfrom log compression
        2. You can perfrom a full contrast stretch (fsimage)
        3. You can take negative (255 - fsimage)
        4. etc.
        """
        a = 0
        b = 255
        min = np.min(image)
        max = np.max(image)
        rows, columns = np.shape(image)
        image1 = np.zeros((rows, columns), dtype=int)
        for i in range(rows):
            for j in range(columns):
                if (max - min) == 0:
                    image1[i, j] = ((b - a) / 0.000001) * (image[i, j] - min)
                else:
                    image1[i, j] = ((b - a) / (max - min)) * (image[i, j] - min)
        return np.uint8(image1)


    def filter(self):
        """Performs frequency filtering on an input image
        returns a filtered image, magnitude of DFT, magnitude of filtered DFT
        ----------------------------------------------------------
        You are allowed to used inbuilt functions to compute fft
        There are packages available in numpy as well as in opencv
        Steps:
        1. Compute the fft of the image
        2. shift the fft to center the low frequencies
        3. get the mask (write your code in functions provided above) the functions can be called by self.filter(shape)
        4. filter the image frequency based on the mask (Convolution theorem)
        5. compute the inverse shift
        6. compute the inverse fourier transform
        7. compute the magnitude
        8. You will need to do post processing on the magnitude and depending on the algorithm (use post_process_image to write this code)
        Note: You do not have to do zero padding as discussed in class, the inbuilt functions takes care of that
        filtered image, magnitude of DFT, magnitude of filtered DFT: Make sure all images being returned have grey scale full contrast stretch and dtype=uint8
        """
        image = self.image
        initial_ft = np.fft.fft2(image)
        initial_shift_fft = np.fft.fftshift(initial_ft)
        magnitude_dft = np.log(np.abs(initial_shift_fft))
        before_dft = self.post_process_image(magnitude_dft)
        mask = self.get_mask(image.shape)
        after_filter_dft = np.multiply(mask, initial_shift_fft)
        mag_filtered_dft = np.log(np.abs(after_filter_dft)+1)
        filtered_dft = self.post_process_image(mag_filtered_dft)
        shift_inverse_ft = np.fft.ifftshift(after_filter_dft)
        ifft = np.fft.ifft2(shift_inverse_ft)
        mag = np.abs(ifft)
        filtered_image = self.post_process_image(mag)
        return [np.uint8(filtered_image),np.uint8(before_dft),np.uint8(filtered_dft),]

