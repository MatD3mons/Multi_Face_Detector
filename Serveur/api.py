import os
from flask import Flask
from flask import request
from flask import render_template
from flask import Response

import numpy as np
import cv2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch
import torchvision.models as models
from Serveur.Thread_Camera import Thread_Camera
#from keras.models import load_model
#import tensorflow as tf
from Serveur.Thread_Camera import Camera

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
pile_boxes = []
pile_image = []
cv2.setUseOptimized(True);
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
#ss = cv2.ximgproc.createStructuredEdgeDetection("model/edge_boxes.yml")


def numpy_to_jpg(numpy):
    ret, jpeg = cv2.imencode('.jpg', numpy)
    frame = jpeg.tobytes()
    return frame

def get_model(num_classes,path):
    # load an object detection model pre-trained on COCO
    model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)  # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new on
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    model.load_state_dict(torch.load(path))
    return model

def face_recognizer(frame,take):
    img = frame
    img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    img = torch.tensor(img.transpose(2, 1, 0)).cuda()
    chanel, width, height = img.shape
    # put the model in evaluation mode
    """
    if take == "RCNN":
        ss.setBaseImage(frame)
        ss.switchToSelectiveSearchFast()
        boxes = ss.process()
        for e, result in enumerate(boxes):
            x, y, w, h = result
            timage = frame[y:y + h, x:x + w]
            resized = cv2.resize(timage, (224, 224), interpolation=cv2.INTER_AREA)
            img = np.expand_dims(resized, axis=0)
            with tf.device('/device:GPU:0'):
                out = RCNN.predict(img)
                if out[0][0] > 0.65:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2, cv2.LINE_AA)
    """
    if take == "FasterRCNN":
        FasterRCNN.eval()

        with torch.no_grad():
            prediction = FasterRCNN([img])

        for element in range(len(prediction[0]["boxes"])):
            boxes = prediction[0]["boxes"][element].cpu().numpy()
            score = np.round(prediction[0]["scores"][element].cpu().numpy(), decimals=4)
            if score > 0.25:
                cv2.rectangle(frame, (int(boxes[1]), int(boxes[0])), (int(boxes[3]), int(boxes[2])), (0, 255, 0), 2, cv2.LINE_AA)

    if take == "cv2":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY);
        faces = CV2.detectMultiScale(gray, 1.3, 5);
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return height, width, frame

# No caching at all for API endpoints.
@app.after_request
def add_header(response):
    # response.cache_control.no_store = True
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

def gen(Camera):
    while True:
        try:
            success, image = Camera.read()
            pile_image.append(image)
            boxes = pile_boxes.pop()
            try:
                for boxe in boxes:
                    cv2.rectangle(image, (int(boxe[1]), int(boxe[0])), (int(boxe[3]), int(boxe[2])), (0, 255, 0), 2,cv2.LINE_AA)
            except:
                pass
            frame = numpy_to_jpg(image)
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        except:
            pass

@app.route('/video_feed')
def video_feed():
    return Response(gen(Camera), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/")
def main():
    Camera.reset()
    return render_template("index.html")

@app.route("/image", methods=["GET", "POST"])
def image():
    Camera.reset()
    if request.method == "POST" and request.files:
        image_file = request.files["image"]
        take = request.form.get('take')
        print(take)
        if image_file:
            full_filename_before = os.path.join('static', 'before.jpg')
            image_file.save(full_filename_before)
            full_filename_after = os.path.join('static', 'after.jpg')
            height, width ,  prediction = face_recognizer(cv2.imread(full_filename_before),take)
            cv2.imwrite(full_filename_after,prediction)
            return render_template("image.html",is_image=1, prediction=full_filename_after, image=full_filename_before)
    return render_template("image.html", prediction = 0, image=None)

@app.route("/camera", methods=["GET", "POST"])
def camera():
    Camera.setup()
    if request.method == "POST":
        return render_template("camera.html")

#first commit
if __name__ == "__main__":
    FasterRCNN = get_model(2,os.path.join("models", "FasterRCNN"))
    CV2 = cv2.CascadeClassifier(os.path.join("models", "haarcascade_frontalface_default.xml"))
    #RCNN = load_model(os.path.join("models", "RCNN.h5"))
    Camera = Camera()
    threader = Thread_Camera(pile_image,pile_boxes,FasterRCNN)
    threader.start()
    app.run(port=12000, debug=True)