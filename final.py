import cv2
import numpy as np
from tkinter import *
from PIL import ImageTk


def viddetect():
    cap = cv2.VideoCapture("test3.mp4")
    whT = 288
    confThreshold = 0.5
    nmsThreshold = 0.4

    classesFile = 'coco.names'
    classNames = []

    with open(classesFile, 'rt') as f:
        classNames = f.read().rstrip('\n').split('\n')

    colors = np.random.uniform(0, 255, size=(len(classNames), 3))
    modelConfiguration = 'yolov3.cfg'
    modelWeights = 'yolov3.weights'

    net = cv2.dnn.readNet(modelConfiguration, modelWeights)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def findObjects(outputs, img):
        hT, wT, cT = img.shape
        bbox = []
        classIds = []
        confs = []

        for output in outputs:
            for det in output:
                scores = det[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > confThreshold:
                    w, h = int(det[2] * wT), int(det[3] * hT)
                    x, y = int((det[0] * wT) - w / 2), int((det[1] * hT) - h / 2)
                    bbox.append([x, y, w, h])
                    classIds.append(classId)
                    confs.append(float(confidence))

        indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)

        for i in indices:
            i = i[0]
            box = bbox[i]
            x, y, w, h = box[0], box[1], box[2], box[3]
            color = colors[classIds[i]]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, f'{classNames[classIds[i]].upper()} {int(confs[i] * 100)}%', (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


    while True:
        success, img = cap.read()
        blob = cv2.dnn.blobFromImage(img, 1 / 255, (whT, whT), [0, 0, 0], 1,
                                     crop=False)
        net.setInput(blob)
        layerNames = net.getLayerNames()
        outputNames = [layerNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        outputs = net.forward(outputNames)
        findObjects(outputs, img)
        cv2.imshow('Detect', img)
        cv2.waitKey(1)


def camdetect():
    cap = cv2.VideoCapture(1)
    whT = 288
    confThreshold = 0.5
    nmsThreshold = 0.4

    classesFile = 'coco.names'
    classNames = []

    with open(classesFile, 'rt') as f:
        classNames = f.read().rstrip('\n').split('\n')

    colors = np.random.uniform(0, 255, size=(len(classNames), 3))
    modelConfiguration = 'yolov3.cfg'
    modelWeights = 'yolov3.weights'

    net = cv2.dnn.readNet(modelConfiguration, modelWeights)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def findObjects(outputs, img):
        hT, wT, cT = img.shape
        bbox = []
        classIds = []
        confs = []

        for output in outputs:
            for det in output:
                scores = det[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > confThreshold:
                    w, h = int(det[2] * wT), int(det[3] * hT)
                    x, y = int((det[0] * wT) - w / 2), int((det[1] * hT) - h / 2)
                    bbox.append([x, y, w, h])
                    classIds.append(classId)
                    confs.append(float(confidence))

        indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)

        for i in indices:
            i = i[0]
            box = bbox[i]
            x, y, w, h = box[0], box[1], box[2], box[3]
            color = colors[classIds[i]]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, f'{classNames[classIds[i]].upper()} {int(confs[i] * 100)}%', (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


    while True:
        success, img = cap.read()
        blob = cv2.dnn.blobFromImage(img, 1 / 255, (whT, whT), [0, 0, 0], 1,
                                     crop=False)
        net.setInput(blob)
        layerNames = net.getLayerNames()
        outputNames = [layerNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        outputs = net.forward(outputNames)
        findObjects(outputs, img)
        cv2.imshow('Detect', img)
        cv2.waitKey(1)


class Show:
    def __init__(self,root):
        self.root = root
        self.root.title("On Road Obstacle Detection")
        self.root.geometry("1199x600+100+50")
        self.root.resizable(False, False)
        self.bg = ImageTk.PhotoImage(file="front.jpg")
        self.bg_image = Label(self.root, image=self.bg).place(x=0, y=0, relwidth=1, relheight=1)
        Frame_show = Frame(self.root, bg="white")
        Frame_show.place(x=150, y=150, height=300, width=700)

        title = Label(Frame_show, text="On Road Obstacle Detection", font=("Impact", 40), fg="#FF5733",
                      bg="white").place(x=40, y=10)
        subtitle = Label(Frame_show, text="Click any of the options below",
                         font=("times new roman", 20, "bold"), fg="#FF5733", bg="white").place(x=40, y=120)
        Show_btn = Button(Frame_show, text="Detect through Video", fg="white", bg="#FF5733", font=("times new roman", 20, "bold"),
                           command=viddetect).place(x=60, y=180)
        Show_btn2 = Button(Frame_show, text="Detect through Camera", fg="white", bg="#FF5733", font=("times new roman", 20, "bold"),
                           command=camdetect).place(x=360, y=180)

root=Tk()
obj=Show(root)
root.mainloop()
