import cv2
import keras
import numpy as np
import matplotlib.pyplot as plt

model = keras.models.load_model("FaceMaskdetector.keras")

cam = cv2.VideoCapture(0)

cv2.namedWindow("test")

img_counter = 0

while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("test", frame)

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1
        img = cv2.imread(img_name)
        img = cv2.resize(img,(128,128),interpolation=cv2.INTER_CUBIC)
        print(model.predict(np.expand_dims(img,0)))

cam.release()

cv2.destroyAllWindows()




