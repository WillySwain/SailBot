import math

import cv2
import numpy as np
import time
import serial

# Checked with TF03,

'''
To run, run the file with the weights, names and config in the same folder.
To exit, press escape.
@author: Sarthak Shrivastava
@author: William Swain
@version: 2021-02-17
'''

ser = serial.Serial("/dev/ttyAMA0", 115200)

net = cv2.dnn.readNet("./yolov3-tiny.weights", "./yolov3-tiny.cfg")


def read_data():
    while True:
        counter = ser.in_waiting  # count the number of bytes of the serial port
        if counter > 8:
            bytes_serial = ser.read(9)
            ser.reset_input_buffer()

            if bytes_serial[0] == 0x59 and bytes_serial[1] == 0x59:  # this portion is for python3
                print("Printing python3 portion")
                distance = bytes_serial[2] + bytes_serial[3] * 256
                print("Distance:" + str(distance))
                if distance <= 30:  # minimum distance before turning on cameras
                    results = detect(0)  # check left camera
                    results2 = detect(1)
                    print(results[0])
                    print(results2[0])
                    if results[0] > results2[0]:
                        print(results[1])
                    else:
                        print(results2[1])
                    time.sleep(3)
                ser.reset_input_buffer()


# classes stores all the categories of objects we can detect
def detect(camera):
    classes = []
    with open("./coco.names", 'r') as f:
        classes = f.read().splitlines()

    # names of the layers goes in layer_names
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # video capture from default computer video source
    cap = cv2.VideoCapture(camera)
    boxes = []
    confidences = []
    class_ids = []
    areas = []
    counter = 0
    while counter < 5:
        # loading image from video source cap
        print(counter)
        _, img = cap.read()
        # storing the height, width and number of channels of original image
        height, width, channels = img.shape
        # making a blob to` run our yolo algorithm on
        blob = cv2.dnn.blobFromImage(img, 1 / 255, (320, 320), (0, 0, 0), True, crop=False)

        net.setInput(blob)
        outs = net.forward(output_layers)
        # forward to outs

        # boxes, confidences and class_ids are an array of the boxes (size of box),
        # confidences (confidence level) and class_ids (type of object) detected
        # for every output layer (out) in our forwarded output_layers (outs)
        for out in outs:
            # for every detection made in each output layer
            for detection in out:
                scores = detection[5:]
                # detection's first few values contain info on height, width, size etc
                # scores contains all the recognition scores for every type of object we can detect
                class_id = np.argmax(scores)
                # class_id stores the name of the class with highest recognition score
                confidence = scores[class_id]
                # confidence stores the highest recognition score value (aka our confidence)
                # we have a very low limit to the confidence minimum (0.3) here, this might be moved up or down later
                if confidence > 0.3:
                    # Object detected.
                    # check distance of detection with lidar
                    # if distance > certain range
                    # continue

                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    areas.append(x * y)
                    # center_x is center of detected object
                    # x is the horizontal space the object occupies
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
                # cv2.rectangle(img, (x,y),(x+w, y+h), (0,255,0), 2)
                # cv2.circle(img, (center_x, center_y), 10, (0, 255, 0), 2)
        cv2.imshow("Image", img)
        k = cv2.waitKey(1)
        counter += 1
        print(len(boxes))

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    # this line is used to supress multiple detections of same object

    font = cv2.FONT_HERSHEY_PLAIN

    i = areas.index(max(areas))
    x, y, w, h = boxes[i]

    label = str(classes[class_ids[i]])
    recognition_score = str(round(confidences[i], 2))
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(img, label, (x, y + 30), font, 2, (255, 255, 255), 2)
    cv2.putText(img, recognition_score, (x, y + 60), font, 2, (255, 255, 255), 2)

    if camera == 1:  # right camera first
        if center_x < width / 2:
            direction = "LEFT"  # object is to the right of the boat
        else:
            direction = "STRAIGHT"
    else:
        if center_x > width / 2:
            direction = "RIGHT"  # the object is to the left of the boat
        else:
            direction = "STRAIGHT"

    cv2.imshow("Image", img)
    ans = [areas[i], direction]

    cap.release()
    cv2.destroyAllWindows()

    return ans
    # if user presses esc break from the loop


read_data()

