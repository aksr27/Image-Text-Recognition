# cd Desktop\Character_recognition
# python char_recognition.py
import matplotlib
matplotlib.use("Agg")
 
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import pickle
import cv2
import os
import json

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset of images")
# ap.add_argument("-m", "--model", required=True,
# 	help="path to output trained model")
# ap.add_argument("-l", "--label-bin", required=True,
# 	help="path to output label binarizer")
# ap.add_argument("-p", "--plot", required=True,
# 	help="path to output accuracy/loss plot")
args = vars(ap.parse_args())

# initialize the data and labels
print("Loading images...")
data = []
labels = []

# grab the image paths and randomly shuffle them
imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.seed(42)
random.shuffle(imagePaths)
print(len(imagePaths))

for imagePath in imagePaths:
	# print(imagePaths)
	image = cv2.imread(imagePath)
	image = cv2.resize(image, (32, 32)).flatten()
	data.append(image)
 
	label = imagePath.split(os.path.sep)[-2]
	labels.append(label)
print(data)
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)
(trainX, testX, trainY, testY) = train_test_split(data,
	labels, test_size=0.05, random_state=42)
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

print("Creating Network...")

model = Sequential()
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(512, activation="relu"))
model.add(Dense(len(lb.classes_), activation="softmax"))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

EPOCHS = 75

print("Training Network....")

model.fit(trainX, trainY, validation_data=(testX, testY),
	epochs=EPOCHS, batch_size=32)

print("[INFO] serializing network and label binarizer...")
model.save("char_model")
f = open("label_bin", "wb")
f.write(pickle.dumps(lb))
f.close()

score = model.evaluate(testX, testY, batch_size=32)
print("Score",score)

print("Evaluating network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1), target_names=lb.classes_))

