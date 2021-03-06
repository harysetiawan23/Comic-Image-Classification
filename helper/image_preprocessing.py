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
        return img_dilate



class HSVPreprocessing:
    def __init__(self,width,heigh,inter=cv2.INTER_AREA):
        # store the target image width, height, and interpolation
		# method used when resizing
        self.width = width
        self.heigh = heigh
        self.inter = inter

        # define range of class threshold color in HSV
        self.babi_threshold = np.array([[169 , 0, 180], [179 ,  127,  244]])
        self.buaya_threshold = np.array([[18 ,0, 114],[59 , 255, 235]])
        self.gajah_threshold = np.array([[0 , 0, 115],[179 ,  51,  179]])
        self.kucing_threshold = np.array([[0 ,0,  89],[17 ,  162,244]])

    def preprocess(self, image):
		# resize the image to a fixed size, ignoring the aspect
		# ratio
        rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

        # Reduce Noice
        noisereduced = cv2.fastNlMeansDenoisingColored(rgb,None,32,32,7,21)

        hsv = cv2.cvtColor(noisereduced,cv2.COLOR_RGB2HSV)

        masked_babi = cv2.inRange(hsv,self.babi_threshold[0],self.babi_threshold[1])
        masked_buaya = cv2.inRange(hsv,self.buaya_threshold[0],self.buaya_threshold[1])
        masked_gajah = cv2.inRange(hsv,self.gajah_threshold[0],self.gajah_threshold[1])
        masked_kucing = cv2.inRange(hsv,self.kucing_threshold[0],self.kucing_threshold[1])

        babi_buaya = cv2.add(masked_babi,masked_buaya)
        gajah_kucing = cv2.add(masked_gajah,masked_kucing,)
        combined = cv2.add(babi_buaya,gajah_kucing)


        selected_image = cv2.bitwise_and(noisereduced,noisereduced,mask=combined)
        selected_image = cv2.resize(selected_image,(self.width,self.heigh),interpolation=self.inter)
        selected_image = cv2.cvtColor(selected_image,cv2.COLOR_RGB2HSV)
        return selected_image