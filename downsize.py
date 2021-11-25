from PIL import Image
import os

imagePath = "./face/face1"
pathList = os.listdir(imagePath)
savePath = "./face/face_downsize1"

for i in pathList:
    path = f'{imagePath}/{i}'
    with Image.open(path) as im:
        im_resized = im.resize((256,256))
        im_resized.save(f'{savePath}/{i}')
