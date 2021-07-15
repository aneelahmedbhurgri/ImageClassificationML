# import the necessary packages
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,	help="path to trained model model")
ap.add_argument("-i", "--image", required=True, help="path to input image")
args = vars(ap.parse_args())

# load the image
image = cv2.imread(args["image"])
orig = image.copy()

# pre-process the image for classification
image = cv2.resize(image, (28, 28))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

# load the trained convolutional neural network
print("[INFO] loading network...")
model = load_model(args["model"])

# classify the input image
(notGalaxy, elliptical, irregular, spiral) = model.predict(image)[0]
#(notGalaxy, galaxy) = model.predict(image)[0]

# build the label
if elliptical > irregular and elliptical > spiral and elliptical > notGalaxy:
    label = "Elliptical Galaxy"
    proba = elliptical
elif irregular > elliptical and irregular > spiral and irregular > notGalaxy:
    label = "Irregular Galaxy"
    proba = irregular
elif spiral > elliptical and spiral > irregular and spiral > notGalaxy:
    label = "Spiral Galaxy"
    proba = spiral
else:
    label = "Not Galaxy"
    proba = notGalaxy
#label = "Galaxy" if galaxy > notGalaxy else "Not Galaxy"
#proba = galaxy if galaxy > notGalaxy else notGalaxy
label = "{}: {:.2f}%".format(label, proba * 100)

# draw the label on the image
output = imutils.resize(orig, width=400)
cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# show the output image
cv2.imshow("Output", output)
cv2.waitKey(0)
