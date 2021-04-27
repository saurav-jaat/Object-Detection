import cv2
import numpy as np 

net=cv2.dnn.readNet('yolov3.weights','yolov3.cfg')

# Extracting Variable names from COCO files and putting them into a list.


classes=[]

with open('coco.names','r') as f:
     classes=f.read().splitlines()


cap = cv2.VideoCapture("test.mp4")
#Loading the image
#img=cv2.imread('front.jpg')

while True:
    _, img = cap.read()
    height, width,_ = img.shape

    # We have to make input image as 416*416 in order to fit in yolo,normalize the pixel value by dividing 255

    #swap RB is for RGB order
    blob=cv2.dnn.blobFromImage(img,1/255,(416,416),(0,0,0),swapRB=True,crop=False)

    net.setInput(blob) #input image into the network

    output_layers_names = net.getUnconnectedOutLayersNames() # Get the output layer names
    layerOutputs= net.forward(output_layers_names) 


    boxes=[] #list to store boudind boxes
    confidences=[]
    class_ids=[] # list to store the predicted classes

    for output in layerOutputs: #extract information from layer outputs
        for detection in output:
            scores = detection[5:] # first element store the probablity 1 and 0 then next 4 stores the bounding box position and 6 to 85 have class probablity
            class_id = np.argmax(scores) # location that contain highes score
            confidence = scores[class_id] # assign into confidence
            if confidence > 0.5:
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)

                x= int(center_x-w/2) # detecting upper left corner for open cv
                y= int(center_y-h/2)

                boxes.append([x,y,w,h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)



    print(len(boxes))

    # NON MAX SUPRESSIOn
    indexes = cv2.dnn.NMSBoxes(boxes,confidences,0.5,0.4)
    #showing details in an image
    font =cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0,255,size=(len(boxes),3))

    #for loop to identify the object detected
    for i in indexes.flatten():
        x,y,w,h=boxes[i] #location and size fo rectangle
        label = str(classes[class_ids[i]]) #getting classes name
        confidence = str(round(confidences[i],2))
        color = colors[i]
        cv2.rectangle(img, (x,y), (x+w ,y+h), color ,2 )
        cv2.putText(img, label + " " + confidence, (x, y+20) ,font ,2 ,(255,255,255), 2)


    cv2.imshow('Image',img)
  
    key = cv2.waitKey(1)
    if key==27:
        break

cap.release()
cv2.destroyAllWindows()

