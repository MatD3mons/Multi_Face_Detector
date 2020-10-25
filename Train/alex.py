import numpy as np
import cv2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch
import torchvision.models as models

def get_model(num_classes):
    # load an object detection model pre-trained on COCO
    model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)  # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new on
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def get_boxe(image):
    # put the model in evaluation mode
    loaded_model.eval()

    with torch.no_grad():
        prediction = loaded_model(image)

    for element in range(len(prediction[0]["boxes"])):
        boxes = prediction[0]["boxes"][element].cpu().numpy()
        score = np.round(prediction[0]["scores"][element].cpu().numpy(), decimals=4)

    return score,boxes

def face_recognizer(path):
    frame = cv2.imread(path)
    img = frame
    img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    img = torch.tensor(img.transpose(2, 1, 0)).cuda()

    # put the model in evaluation mode
    loaded_model.eval()

    with torch.no_grad():
        prediction = loaded_model([img])

    for element in range(len(prediction[0]["boxes"])):
        boxes = prediction[0]["boxes"][element].cpu().numpy()
        score = np.round(prediction[0]["scores"][element].cpu().numpy(), decimals=4)
        if score > 0.15:
            cv2.rectangle(frame, (boxes[1], boxes[0]), (boxes[3], boxes[2]), (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("preview", frame)
    cv2.waitKey(0)

def webcam_face_recognizer():
    cv2.namedWindow("preview")
    vc = cv2.VideoCapture(0)
    counter = 0;
    while vc.isOpened():
        i, frame = vc.read()
        img = frame
        img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        img = torch.tensor(img.transpose(2, 1, 0)).cuda()

        # put the model in evaluation mode
        loaded_model.eval()

        with torch.no_grad():
            prediction = loaded_model([img])

        for element in range(len(prediction[0]["boxes"])):
            boxes = prediction[0]["boxes"][element].cpu().numpy()
            score = np.round(prediction[0]["scores"][element].cpu().numpy(), decimals=4)
            cv2.rectangle(frame, (boxes[1], boxes[0]), (boxes[3], boxes[2]), (0, 255, 0), 2, cv2.LINE_AA)
            break

        key = cv2.waitKey(100)
        cv2.imshow("preview", frame)

        if key == 27:  # exit on ESC
            break
    cv2.destroyWindow("preview")


if __name__ == "__main__":
    loaded_model = get_model(num_classes=2)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    loaded_model.to(device)
    loaded_model.load_state_dict(torch.load("model"))
    #webcam_face_recognizer()
    face_recognizer("valide/7   .jpg")