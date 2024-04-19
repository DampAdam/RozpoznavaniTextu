import PIL.Image
from tools import *
import pytesseract as pt
import numpy as np
from PIL import Image

def main():
    clr()

    model = inference.get_model("dpdtextrecognition/3")
    # pt.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

    Image.open(r'C:\Users\badam\Documents\RozpoznavaniTextuDataset\Obrázek WhatsApp 20231025 v 100021_2d76acec.jpg').show()

    img = cv2.imdecode(np.fromfile(r'C:\Users\badam\Documents\RozpoznavaniTextuDataset\Obrázek WhatsApp 20231025 v 100021_2d76acec.jpg', dtype=np.uint8), cv2.IMREAD_COLOR)

    #cv2.resizeWindow('Image', img.shape[1], img.shape[0])
    prediction = getPhonePoints(img)

    cv2.resizeWindow('frame', prediction.shape[1], prediction.shape[0])
    cv2.imshow('frame', prediction)


    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()