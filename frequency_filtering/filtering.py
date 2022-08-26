

import numpy as np


class Filtering:

    def __init__(self, image):

        self.image = image
        self.mask = self.get_mask

    def get_mask(self, shape):

        mask = np.ones(shape)
        mask[225:245, 265:285] = 0
        mask[265:285,220:240]=0
        mask[225:235, 55:68] = 0
        mask[450:460, 275:295] = 0
        mask[272:289, 448:463] = 0
        mask[55:68, 220:240] = 0
        return mask

    def post_process_image(self, image):

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

