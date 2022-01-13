import cv2
import numpy as np
from PIL import Image
import os

facePath = "./face/face_downsize5"
comicsPath = "./comics/comics_downsize5"

faceList = os.listdir(facePath)

for i in faceList:
    faceImg = cv2.cvtColor(cv2.imread(f"{facePath}/{i}"),cv2.COLOR_BGR2RGB)
    comicsImg = cv2.cvtColor(cv2.imread(f"{comicsPath}/{i}"),cv2.COLOR_BGR2RGB)
    img = np.concatenate((comicsImg,faceImg),axis=1)
    img = Image.fromarray(img)
    img.save(f"./comcact_image/5/{i}")
