from threading import Thread

import cv2
import torch
import numpy as np
import time

class Thread_Camera(Thread):

    def __init__(self,pile_image,pile_boxes,model):
        Thread.__init__(self)
        self.pile_image = pile_image
        self.pile_boxes = pile_boxes
        self.model = model

    def run(self):
        while True:
            boxes = []
            while len(self.pile_image):
                img = self.pile_image.pop()
            try:
                img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                img = torch.tensor(img.transpose(2, 1, 0)).cuda()
                # put the model in evaluation mode
                self.model.eval()

                with torch.no_grad():
                    prediction = self.model([img])

                for element in range(len(prediction[0]["boxes"])):
                    boxe = prediction[0]["boxes"][element].cpu().numpy()
                    score = np.round(prediction[0]["scores"][element].cpu().numpy(), decimals=3)
                    if score > 0.5:
                        boxes.append(boxe)
                self.pile_boxes.append(boxes)
            except:
                pass

class Camera():

    def __init__(self):
        self.cap = None

    def setup(self):
        self.cap = cv2.VideoCapture(0)

    def read(self):
        try:
            return self.cap.read()
        except:
            pass

    def reset(self):
        try:
            self.cap.release()
        except:
            pass