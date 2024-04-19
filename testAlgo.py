import functools as ft
import time
from matplotlib.widgets import Slider
import pytesseract as pt
from tools import *
from threading import Thread
from queue import Queue
from queue import Empty
import pytesseract as pt

def main():
    clr()

    def showLiveFeed(threadname, q):
        # http://192.168.250.246:8080/video
        cap = cv2.VideoCapture(0)
        cv2.VideoCapture.set(cap, cv2.CAP_PROP_FRAME_WIDTH, 1920)
        while True:
            _, frame = cap.read()
            try:
                filters = q.get_nowait()
            except Empty:
                pass


            frame, found, confidence = cropToPhone(frame)
            og = frame.copy()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if found:
                results1 = getTextVision(frame).text_annotations

                frame = cv2.bilateralFilter(frame, filters['bilateralFilter']['d'], filters['bilateralFilter']['sigmaColor'], filters['bilateralFilter']['sigmaSpace'])
                frame = cv2.Canny(frame, filters['canny']['threshold1'], filters['canny']['threshold2'])
                frame = cv2.dilate(frame, (filters['dilate']['kernel'], filters['dilate']['kernel']), iterations=filters['dilate']['iterations'])
                frame = cv2.erode(frame, (filters['erode']['kernel'], filters['erode']['kernel']), iterations=filters['erode']['iterations'])
                zob = frame.copy()

                contours, _ = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                contours = filterContoursByArea(contours, 100)
                contours = getBiggestContours(contours, 1)
                #contours = approximateContours(contours, filters['approxPolyDP']['epsilon'])
                #cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)
                contours = getMinAreaRectangles(contours)
                # drawRectanglesFromContours(frame, contours)
                # drawRectangles(frame, contours)
                if len(contours) != 0:
                    p1, p2, _ = contours[0]
                    print(p1)
                    print(p2)
                    cv2.rectangle(frame, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0, 255, 0), 2)
                    # frame = warpToRectangle(og, contours[0])
                    frame = cv2.flip(frame, 1)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    # frame = zob
                    # clr()
                    # print(pt.image_to_string(frame, config=f'--psm {filters["pytesseract"]["psm"]} --oem {filters["pytesseract"]["oem"]}'))
                    pass
                #frame = autoBrighten(frame)
                frame = autoConstrast(frame)
                results2 = getTextVision(frame).text_annotations
                for result in results2:
                    verticies, description = result.bounding_poly, result.description
                    cv2.putText(frame, description, (verticies.vertices[0].x, verticies.vertices[0].y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.rectangle(frame, (verticies.vertices[0].x, verticies.vertices[0].y), (verticies.vertices[2].x, verticies.vertices[2].y), (255, 255, 255), 2)

                clr()
                print(results1)
                if results1 != [] and results2 != []:
                    print(len(results1))
                    print(f'Original: {results1[0].description}\nProcessed: {results2[0].description}')
                cv2.imshow('frame', zob)
                time.sleep(0.5)
            else:
                #frame = autoBrighten(frame)
                cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()


    q = Queue()
    thread1 = Thread(target=showLiveFeed, args=('liveFeed', q))
    #thread2 = Thread(target=process, args=('process', queue))
    #thread2.start()
    #thread1.join()
    #thread2.join()

    filters = {
        'bilateralFilter': {
            'd': 1,
            'sigmaColor': 100,
            'sigmaSpace': 100
        },
        'canny': {
            'threshold1': 0,
            'threshold2': 0
        },
        'dilate': {
            'kernel': 1,
            'iterations': 5
        },
        'erode': {
            'kernel': 1,
            'iterations': 2
        },
        'approxPolyDP': {
            'epsilon': 0.02
        },
        'pytesseract': {
            'psm': 6,
            'oem': 3
        }
    }

    maxValues = {
        'bilateralFilter': {
            'd': 100,
            'sigmaColor': 1000,
            'sigmaSpace': 1000
        },
        'canny': {
            'threshold1': 255,
            'threshold2': 255
        },
        'dilate': {
            'kernel': 10,
            'iterations': 10
        },
        'erode': {
            'kernel': 10,
            'iterations': 10
        },
        'approxPolyDP': {
            'epsilon': 0.1
        },
        'pytesseract': {
            'psm': 13,
            'oem': 3
        }
    }

    minValues = {
        'bilateralFilter': {
            'd': 1,
            'sigmaColor': 0,
            'sigmaSpace': 0
        },
        'canny': {
            'threshold1': 0,
            'threshold2': 0
        },
        'dilate': {
            'kernel': 1,
            'iterations': 0
        },
        'erode': {
            'kernel': 1,
            'iterations': 0
        },
        'approxPolyDP': {
            'epsilon': 0
        },
        'pytesseract': {
            'psm': 0,
            'oem': 0
        }
    }

    stepValues = {
        'bilateralFilter': {
            'd': 1,
            'sigmaColor': 1,
            'sigmaSpace': 1
        },
        'canny': {
            'threshold1': 1,
            'threshold2': 1
        },
        'dilate': {
            'kernel': 1,
            'iterations': 1
        },
        'erode': {
            'kernel': 1,
            'iterations': 1
        },
        'approxPolyDP': {
            'epsilon': 0.01
        },
        'pytesseract': {
            'psm': 1,
            'oem': 1
        }
    }

    q.put(filters)
    thread1.start()

    def update(filter, param, value):
        filters[filter][param] = value
        q.put(filters)
    fig, ax = plt.subplots(len(filters), len(filters['bilateralFilter']), figsize=(10, 10))
    sliders = []
    for i, (filter, params) in enumerate(filters.items()):
        for j, (param, value) in enumerate(params.items()):
            slider = Slider(ax=ax[i, j], label=f"{filter}:{param}", valmin=minValues[filter][param], valmax=maxValues[filter][param], valstep=stepValues[filter][param], valinit=value)
            sliders.append(slider)
            slider.on_changed(ft.partial(update, filter, param))
            update(filter, param, value)
    plt.show()

if __name__ == '__main__':
    main()