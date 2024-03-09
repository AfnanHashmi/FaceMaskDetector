import cv2
import glob
import keras
import numpy
import numpy as np

model = keras.models.load_model("FaceMaskdetector.keras")
model.summary()

dataset = []

afiles = r"boy-5749597_1280.jpg"
bfiles = r"download.jpeg"

img1 = cv2.imread(afiles)
img1 = cv2.resize(img1,(128,128))
img2 = cv2.imread(bfiles)
img2 = cv2.resize(img2,(128,128))


img1 = np.expand_dims(img1,0)
img2 = np.expand_dims(img2,0)

print(model.predict(img1))
print(model.predict(img2))