import cv2
import numpy as np
import tkinter as tk

from keras.models import load_model
from keras.utils import load_img, img_to_array
from tkinter.filedialog import askopenfilename

root = tk.Tk()
root.withdraw()
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# model_path = askopenfilename(title='Select an model')
image_path = askopenfilename(title='Select an image')

model = load_model('models/vgg16-with-hyperband.h5')

pred_image = load_img(image_path, target_size=(218, 178))
pred_image = img_to_array(pred_image) / 255.0
pred_image = np.expand_dims(pred_image, axis=0)

prediction = model.predict(pred_image)

image = cv2.imread(image_path)
gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
faces = face_cascade.detectMultiScale(gray_image, 1.3, 5)
for (x, y, w, h) in faces:
    if np.round(prediction[0][0]) == 0:
        prediction_label = 'Female'
        prediction_accuracy = 1 - prediction[0][0]
        cv2.rectangle(image, (x, y), ((x + w), (y + h)), (0, 255, 0), 2)
        cv2.putText(image, f'Prediction: {prediction_label}',
                    (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(image, f'Accuracy: {prediction_accuracy * 100:.2f}%',
                    (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    else:
        prediction_label = 'Male'
        prediction_accuracy = prediction[0][0]
        cv2.rectangle(image, (x, y), ((x + w), (y + h)), (0, 255, 0), 2)
        cv2.putText(image, f'Prediction: {prediction_label}',
                    (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(image, f'Accuracy: {prediction_accuracy * 100:.2f}%',
                    (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

cv2.imshow("Prediction Result", image)
cv2.waitKey(0)