import os
import pickle

import cv2
import numpy as np

image_dir = 'C:\\Users\Ritav\PycharmProjects\Hydro-Infinity\\venv\Project\ImagesDataset'

face_cascade = cv2.CascadeClassifier(
    'C:\\Users\Ritav\PycharmProjects\Hydro-Infinity\\venv\Project\haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

current_id = 0
label_ids = {}
y_labels = []
x_train = []

for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root, file)
            label = os.path.basename(root).replace(" ", "-").lower()

            if not label in label_ids:
                label_ids[label] = current_id
                current_id += 1
            id_ = label_ids[label]

            image_array = cv2.imread(path)
            gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(gray)

            for (x, y, w, h) in faces:
                roi = gray[y:y + h, x:x + w]

                x_train.append(roi)
                y_labels.append(id_)
# print(y_labels)
# print(x_train)
print(label_ids)

with open("face-labels.pickle", 'wb') as f:
    pickle.dump(label_ids, f)

recognizer.train(x_train, np.array(y_labels))
recognizer.save("face-trainner.yml")
