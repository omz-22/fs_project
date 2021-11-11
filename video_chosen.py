# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 18:34:42 2021

@author: davis
"""

# opencv object tracking
# object detection and tracking opencv
import cv2
import numpy as np
import imutils
import time
import sys
from datetime import timedelta
 
# Load Yolo
yolo_weight = "yolov3-custom_last.weights"
yolo_config = "yolov3.cfg"
coco_labels = "custom.names"
net = cv2.dnn.readNet(yolo_weight, yolo_config)
net = cv2.dnn.readNet(yolo_weight, yolo_config)
classes = []
with open(coco_labels, "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))
starting_time = time.time()
img_id = 0

# Defining desired shape
fWidth = 640
fHeight = 480
 
# Below function will read video frames
cap = cv2.VideoCapture('video-big.mp4')
writer = None
fps = cap.get(cv2.CAP_PROP_FPS)
timestamps = [cap.get(cv2.CAP_PROP_POS_MSEC)]
calc_timestamps = [0.0]


while True:
    read_ok, img = cap.read()
    img_id += 1
    
    if read_ok:
        timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC))
        calc_timestamps.append(calc_timestamps[-1] + 1000/fps)
    else:
        break 
    
    height, width, channels = img.shape
    
    #width  = int(cap.get(640)) # float
    #height = int(cap.get(480)) # float
        
    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (480, 480), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
 
    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    
    
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.6:
                
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
 
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
 
    font = cv2.FONT_HERSHEY_DUPLEX
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence_label = int(confidences[i] * 100)
            color = colors[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, f'{label, confidence_label}', (x-25, y + 75), font, 1, color, 2)
            
            #print('Class_ID ' + 'Confidence ' + 'Timestamp')
   
            tempo =  cap.get(cv2.CAP_PROP_POS_MSEC)
            #video = cap.get(cv2.CAP_PROP_POS_AVI_RATIO)
            mil = 1000
            sec = round((tempo/mil), 2)
            
            #difference = timestamps - calc_timestamps
            
            human_time = timedelta(seconds=sec)
            #timedelta(0, sec)
            
            
    img = imutils.resize(img, width=800)
    elapsed_time = time.time() - starting_time
    fps = img_id / elapsed_time
    cv2.putText(img, "FPS: " + str(round(fps, 2)), (40, 670), font, .7, (0, 255, 255), 1)
    cv2.putText(img, "press [esc] to exit", (40, 690), font, .45, (0, 255, 255), 1)
    
    #for i, (ts, cts) in enumerate(zip(timestamps, calc_timestamps)):
        #print('Frame %d difference:'%i, abs(ts - cts))
    
    #for i, (ts, cts) in enumerate(zip(timestamps, calc_timestamps)):
       # print((label) + ' ' + str(confidence_label) + ' ' + str(sec), abs(ts - cts))
    
      
    
    #print(str(video))
    
    sys.stdout = open('results.csv', mode = 'a')
    
    
    print((label) + ' ' + str(confidence_label) + ' ' + str(human_time))
    
    if writer is None:
        #fourcc = cv2.VideoWriter_fourcc(*'H264')
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        
        writer = cv2.VideoWriter( 'output.mp4', fourcc, 20,
                                 (img.shape[1], img.shape[0]), True)
                                 
    
    cv2.imshow("Image", img)
    writer.write(img)   
    
    key = cv2.waitKey(1)
    if key == 27:
        print("[button pressed] ///// [esc].")
        print("[feedback] ///// Videocapturing succesfully stopped")
        break

    
    sys.stdout.close()
    
writer.release()

#img.release()
    
cv2.destroyAllWindows()
    
    
    