import cv2
import os
import numpy as np


class DatasetLoader:
    def __init__(self, preprocessors=None):
        self.preprocessors = preprocessors

        try:
            if self.preprocessors is None:
                self.preprocessors = []
        except:
            print(type(self.preprocessors))
            self.preprocessors = []

    def load(self, imagePaths, verbose=1):
        # initialize the list of features and labels
        data = []
        labels = []

        for (i, imagePath) in enumerate(imagePaths):
            # load the image and extract the class label assuming
            # that our path has the following format:
            # /path/to/dataset/{class}/{image}.jpg
            image = cv2.imread(imagePath,flags=cv2.IMREAD_COLOR)
            label = imagePath.split(os.path.sep)[-2]

            # check to see if our preprocessors are not None
            if self.preprocessors is not None:
                # loop over the preprocessors and apply each to
                # the image
                for p in self.preprocessors:
                    image = p.preprocess(image)

            # treat our processed image as a "feature vector"
            # by updating the data list followed by the labels
            data.append(image)
            labels.append(label)

            # show an update every `verbose` images
            if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
                print("[INFO] processed {}/{}".format(i + 1,
                                                        len(imagePaths)))

        # return a tuple of the data and labels
        return (np.array(data), np.array(labels))
