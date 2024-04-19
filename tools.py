from ultralytics import YOLO
import os
import numpy as np
import cv2
import functools as ft
import json
import cv2
import os
import matplotlib.pyplot as plt
from google.cloud import vision
from PIL import Image, ImageEnhance, ImageOps

clr = lambda:os.system("cls")

clr()

# ./runs/detect/train6/weights/best.pt

model = YOLO("./runs/detect/train6/weights/best.pt")

def processVideo(func):
    cap = cv2.VideoCapture(0)
    cap.set(3, 1920)
    while True:
        _, frame = cap.read()
        frame = func(frame)
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def getPhonesFromImage(img, classId = 0, confThreshold: float = 0.75):
    results = model(img, verbose=False)[0].boxes.cpu().numpy()
    return [[results.xyxy[i].astype(int), int(1000*results.conf[i])/1000] for i in range(len(results.cls)) if results.conf[i] > confThreshold and results.cls[i] == classId]

clr = lambda:os.system('cls')

getPredictions = lambda img, classId = 0, confThreshold = 0.75: getPhonesFromImage(img, classId, confThreshold)

client = vision.ImageAnnotatorClient()

def getBestPrediction(img, classId = 0, confThreshold = 0.75):
    max_confidence = 0
    best_prediction = None
    predictions = getPredictions(img, classId, confThreshold)
    for (i, prediction) in enumerate(predictions):
        if prediction[1] > max_confidence:
            max_confidence = prediction[1]
            best_prediction = i
    return None if best_prediction == None else predictions[best_prediction]

def cropToPhone(img, classId = 0, confThreshold = 0.75):
    prediction = getBestPrediction(img, classId, confThreshold)
    if prediction == None:
        return (img, False, None)
    x0, y0, x1, y1 = prediction[0]
    return (img[y0:y1, x0:x1], True, prediction[1])

def warpToRectangle(image, rect):
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    # Determine the rectangle's orientation by comparing its width and height
    width, height = rect[1]
    if width > height:
        # Rectangle is horizontal
        src_pts = np.array([box[1], box[2], box[3], box[0]], dtype="float32")
    else:
        # Rectangle is vertical
        src_pts = np.array([box[0], box[1], box[2], box[3]], dtype="float32")
    
    if src_pts[0][1] > src_pts[1][1]:
    # It seems points are ordered bottom-to-top, let's reorder them
        src_pts = np.array([src_pts[3], src_pts[2], src_pts[1], src_pts[0]])
        pass

    # The desired output dimensions (output is always vertical)
    output_height = max(image.shape[0], image.shape[1])
    output_width = min(image.shape[0], image.shape[1])

    # Define the destination points for a vertical output
    dst_pts = np.array([
        [0, 0],
        [0, output_height],
        [output_width, output_height],
        [output_width, 0]
    ], dtype="float32")

    # Compute the perspective transform matrix and apply it
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(image, M, (output_width, output_height))
    return warped

def getPhonePoints(img):
    img = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.bilateralFilter(img, 9, 75, 75)
    cv2.imshow('gray', img)
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    plt.plot(hist, color='gray')
    plt.xlim([0, 256])
    img = cv2.Canny(img, 35, 50)
    cv2.imshow('edged', img)
    img = cv2.dilate(img, (3, 3), iterations=1)
    img = cv2.erode(img, (3, 3), iterations=1)
    cv2.imshow('final', img)
    plt.show()
    return img

filterContoursByArea = lambda contours, area: [contour for contour in contours if cv2.contourArea(contour) > area]

def drawRectanglesFromContours(img, contours):
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return img

def drawRectangles(img, rectangles):
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img
    for rectangle in rectangles:
        box = cv2.boxPoints(rectangle)
        box = np.int0(box)
        cv2.drawContours(img, [box], 0, (0, 255, 0), 2)
    return img

getBiggestContours = lambda contours, n: sorted(contours, key=cv2.contourArea, reverse=True)[:n]

approximateContours = lambda contours, ratio: [cv2.approxPolyDP(contour, ratio*cv2.arcLength(contour, True), True) for contour in contours]

getMinAreaRectangles = lambda contours: [cv2.minAreaRect(contour) for contour in contours]

def getTextVision(img):
    success, encoded_image = cv2.imencode('.jpg', img)
    content2 = encoded_image.tobytes()
    image_cv2 = vision.Image(content=content2)
    response =  client.text_detection(image=image_cv2, image_context={"language_hints": ["cs"]})
    return response

def autoBrighten(img):
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img
    image_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    image_yuv[:,:,0] = cv2.equalizeHist(image_yuv[:,:,0])
    return cv2.cvtColor(image_yuv, cv2.COLOR_YUV2BGR)

autoConstrast = lambda img: np.array(ImageOps.autocontrast(Image.fromarray(img)))

def getPhoneRectangle(frame, classId = 0, confThreshold = 0.75):
    frame, found, _ = cropToPhone(frame, classId, confThreshold)
    # og = frame.copy()
    if found:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # results1 = getTextVision(frame).text_annotations

        frame = cv2.bilateralFilter(frame, 1, 100, 100)
        frame = cv2.Canny(frame, 0, 0)
        frame = cv2.dilate(frame, (1, 1), iterations=5)
        frame = cv2.erode(frame, (1, 1), iterations=2)
        # zob = frame.copy()

        contours, _ = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        contours = filterContoursByArea(contours, 100)
        contours = getBiggestContours(contours, 1)
        #contours = approximateContours(contours, filters['approxPolyDP']['epsilon'])
        #cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)
        contours = getMinAreaRectangles(contours)
        # drawRectanglesFromContours(frame, contours)
        # drawRectangles(frame, contours)
        return (contours[0], frame)
        if len(contours) != 0:
            frame = warpToRectangle(og, contours[0])
            frame = cv2.flip(frame, 1)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # frame = zob
            # clr()
            # print(pt.image_to_string(frame, config=f'--psm {filters["pytesseract"]["psm"]} --oem {filters["pytesseract"]["oem"]}'))
            # pass
        #frame = autoBrighten(frame)
        frame = autoConstrast(frame)
        results2 = getTextVision(frame).text_annotations
        for result in results2:
            verticies, description = result.bounding_poly, result.description
            cv2.putText(frame, description, (verticies.vertices[0].x, verticies.vertices[0].y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.rectangle(frame, (verticies.vertices[0].x, verticies.vertices[0].y), (verticies.vertices[2].x, verticies.vertices[2].y), (255, 255, 255), 2)

        # clr()
        # print(results1)
        # if results1 != [] and results2 != []:
        #     print(len(results1))
        #     print(f'Original: {results1[0].description}\nProcessed: {results2[0].description}')
    return None
