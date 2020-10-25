import os
from flask import Flask
from flask import request
from flask import render_template

import fastai
import torch

from fastai.vision.widgets import *
from fastai.vision.all import *

import cv2
from PIL import Image
import numpy as np
import io

app = Flask(__name__)

path=Path('face_age')

def to_num(x:str):
    return int(x)

get_y= Pipeline([parent_label,to_num])
dblock=DataBlock(blocks=[ImageBlock,RegressionBlock()],
                 get_items=get_image_files,
                 splitter=RandomSplitter(),
                 get_y=get_y,
                 item_tfms=Resize(128, method='squish'),
                batch_tfms=[*aug_transforms(size=64, max_warp=0, max_rotate=7.0, max_zoom=1.0)])

dls=dblock.dataloaders(path,bs=64,verbose=True,num_workers=0 )
learn=cnn_learner(dls,resnet50,loss_func=MSELossFlat(), y_range=(10.0,70.0))
learn.load('TesT')

def get_predict(img):
    pred,_,_=learn.predict(img)
    return pred

@app.route("/", methods=["GET","POST"])
def upload_predict():
    if request.method == "POST":
        image_file = request.files["image"]
        if image_file:
            full_filename = os.path.join('static', image_file.filename)
            image_file.save(full_filename)
            img = PILImage.create(image_file)
            prediction = get_predict(img)
            return render_template("index.html", prediction=prediction, image=full_filename)
    return render_template("index.html", prediction = 0, image=None)

if __name__ == "__main__":
    app.run(port=12000, debug=False)
