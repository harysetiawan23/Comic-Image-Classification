import cv2
import numpy as np
import os


class ImagePreprocessing:
    def __init__(self,width,heigh,inter=cv2.INTER_AREA):
        # store the target image width, height, and interpolation
		# method used when resizing
        self.width = width
        self.heigh = heigh
        self.inter = inter

    def preprocess(self, image):
		# resize the image to a fixed size, ignoring the aspect
		# ratio
        image = cv2.resize(image,(self.width,self.heigh),interpolation=self.inter)

        # Convert to RGB
        rgb_image = cv2.cvtColor(image,code=cv2.COLOR_BGR2RGB)

        # Reduce Noice
        noisereduced = cv2.fastNlMeansDenoisingColored(rgb_image,None,32,32,7,21)

        # Blur
        blur = cv2.GaussianBlur(noisereduced,(5,5),100)

        # Dilatasi
        kernel = np.ones((3, 3), 'uint8')
        img_dilate = cv2.dilate(blur, kernel, iterations=5)

        # Erosi
        img_erosion = cv2.erode(img_dilate, kernel, iterations=15)

        # # Deteksi Tepi dengan cany
        # edges = cv2.Canny(img_dilate,100,200)
        return img_erosion