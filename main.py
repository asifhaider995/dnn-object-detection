import numpy as np
import cv2
import os
from facenet_pytorch import MTCNN
import torch

def detect_faces(img):
    frame = cv2.resize(img, (640, 360))

    #Here we are going to use the facenet detector
    boxes, conf = mtcnn.detect(frame)
    # detections = detect(frame)
    print(boxes)
    print(conf)
    # if conf[0] !=  None:
    #     for (x, y, w, h) in boxes:
    #         text = f"{conf[0]*100:.2f}%"
    #         x, y, w, h = int(x), int(y), int(w), int(h)
    #         cv2.putText(frame, text, (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1,(250, 0, 0), 1)
    #         cv2.rectangle(frame, (x, y), (w, h), (255, 0, 0), 2)
    
    return frame
    # print(boxes)


def detect_objects(image, model_path, proto_path, conf):
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
        "sofa", "train", "tvmonitor"]
    # CLASSES = ["background", "bicycle", "bus", "car", "motorbike", "person", "tvmonitor"]

    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

    print("[INFO] loading model...")
    net = cv2.dnn.readNetFromCaffe(proto_path, model_path)

    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (1920, 1080)), 0.007833, (388, 407), 127.5)

    # pass the blob through the network and obtain the detections and
    # predictions
    print("[INFO] computing object detections...")
    net.setInput(blob)
    detections = net.forward()
    # print(detections)

    # loop over the detections
    for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]
        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > conf:
            # extract the index of the class label from the `detections`,
            # then compute the (x, y)-coordinates of the bounding box for
            # the object
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            # display the prediction
            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
            print("[INFO] {}".format(label))
            if CLASSES[idx] == "person":
                cv2.rectangle(image, (startX, startY), (endX, endY),COLORS[idx], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(image, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
                # image = detect_faces(image[startX: startY, endX: endY])
    return image

def main():
    path = os.getcwd()
    DATA_PATH = os.path.join(path, "data")
    IMG_PATH = os.path.join(DATA_PATH, "dataset", "train")
    model_path = os.path.join(DATA_PATH, "pre-trained")

    caffe_model_path = os.path.join(model_path, os.listdir(model_path)[0])
    deploy_proto_path = os.path.join(model_path, os.listdir(model_path)[1])
    _confidence = 0.45

    print(caffe_model_path)
    print(deploy_proto_path)
    print(_confidence)


    
    # for im in os.listdir(IMG_PATH):
    im = os.listdir(IMG_PATH)
    print(len(im))
    for i in range(4,54):
        image = cv2.imread(os.path.join(IMG_PATH, im[i]))
        # cv2.imshow("DD", image)
        img = detect_objects(image, caffe_model_path, deploy_proto_path, _confidence)
        # img = detect_faces(image)
        # show the output image
        
        cv2.imshow("Output", cv2.resize(img, (1024, 576)))
        # cv2.waitKey(0)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #Create the model
    mtcnn = MTCNN(keep_all=True, device=device)
    main()