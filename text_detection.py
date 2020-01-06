from imutils.object_detection import non_max_suppression
import numpy as np
import argparse
import time
import cv2
# import copy

ap = argparse.ArgumentParser()
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
orig = image.copy()
# orig=cv2.resize(orig,(528, 528), interpolation = cv2.INTER_AREA)
# cv2.imshow('aksr',orig)
(H, W) = image.shape[:2]

# set the new width and height and then determine the ratio in change
# for both the width and height
(newW, newH) = (args["width"], args["height"])
rW = W / float(newW)
rH = H / float(newH)

# resize the image and grab the new image dimensions
image = cv2.resize(image, (newW, newH))
(H, W) = image.shape[:2]

# define the two output layer names for the EAST detector model that
# we are interested -- the first is the output probabilities and the
# second can be used to derive the bounding box coordinates of text
layerNames = [
	"feature_fusion/Conv_7/Sigmoid",
	"feature_fusion/concat_3"]

# load the pre-trained EAST text detector
print("[INFO] loading EAST text detector...")
net = cv2.dnn.readNet(args["east"])

# construct a blob from the image and then perform a forward pass of
# the model to obtain the two output layer sets
blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
	(123.68, 116.78, 103.94), swapRB=True, crop=False)
start = time.time()
net.setInput(blob)
(scores, geometry) = net.forward(layerNames)
end = time.time()

# show timing information on text prediction
print("[INFO] text detection took {:.6f} seconds".format(end - start))

# grab the number of rows and columns from the scores volume, then
# initialize our set of bounding box rectangles and corresponding
# confidence scores
(numRows, numCols) = scores.shape[2:4]
rects = []
confidences = []

# loop over the number of rows
for y in range(0, numRows):
	# extract the scores (probabilities), followed by the geometrical
	# data used to derive potential bounding box coordinates that
	# surround text
	scoresData = scores[0, 0, y]
	xData0 = geometry[0, 0, y]
	xData1 = geometry[0, 1, y]
	xData2 = geometry[0, 2, y]
	xData3 = geometry[0, 3, y]
	anglesData = geometry[0, 4, y]

	# loop over the number of columns
	for x in range(0, numCols):
		# if our score does not have sufficient probability, ignore it
		if scoresData[x] < args["min_confidence"]:
			continue

		# compute the offset factor as our resulting feature maps will
		# be 4x smaller than the input image
		(offsetX, offsetY) = (x * 4.0, y * 4.0)

		# extract the rotation angle for the prediction and then
		# compute the sin and cosine
		angle = anglesData[x]
		cos = np.cos(angle)
		sin = np.sin(angle)

		# use the geometry volume to derive the width and height of
		# the bounding box
		h = xData0[x] + xData2[x]
		w = xData1[x] + xData3[x]

		# compute both the starting and ending (x, y)-coordinates for
		# the text prediction bounding box
		endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
		endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
		startX = int(endX - w)
		startY = int(endY - h)

		# add the bounding box coordinates and probability score to
		# our respective lists
		rects.append((startX, startY, endX, endY))
		confidences.append(scoresData[x])

# apply non-maxima suppression to suppress weak, overlapping bounding
# boxes
boxes = non_max_suppression(np.array(rects), probs=confidences)


import os

# define the name of the directory to be created
path = os.getcwd()+"/cropped"

try:
    os.mkdir(path)
except OSError:  
    print ("Creation of the directory %s failed" % path)
else:  
    print ("Successfully created the directory %s " % path)


# loop over the bounding boxes
i=1
for (startX, startY, endX, endY) in boxes:
	# scale the bounding box coordinates based on the respective
	# ratios
	startX = int(startX * rW)
	startY = int(startY * rH)
	endX = int(endX * rW)
	endY = int(endY * rH)

	# draw the bounding box on the image
	# cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)
	crop_img=orig[startY:endY,startX:endX]
	gray_image = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
	# blur = cv2.bilateralFilter(gray_image,3,31,31)
	blur=cv2.GaussianBlur(gray_image,(5,5),0)
	# edged = cv2.Canny(blur, 30, 180)
	th = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,29,9)
	_, contours, _ = cv2.findContours(th.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	# cv2.rectangle(edged, (0, 0), (1, 3), (0, 255, 0), 2)
	# cv2.drawContours(gray_image, contours, -1, (0,0,255), 1)
	
	h_list=[]
	for cnt in contours:
		x,y,w,h = cv2.boundingRect(cnt)
		if(w>7 and h>13 and w<45 and h<55 and w*h<1500):
			h_list.append([x,y,w,h])
	print(h_list)
	ziped_list=list(zip(*h_list))
	x_list=list(ziped_list[0])
	dic=dict(zip(x_list,h_list))
	x_list.sort()
	for x in x_list:
		[x,y,w,h]=dic[x]
		cv2.rectangle(orig,(startX+x-1,startY+y-1),(startX+x+w+1,startY+y+h+2),(0,255,0),1)
		crop_chr=crop_img[y:y+h,x:x+w]
		cv2.imshow("Text Detection", orig)
		cv2.waitKey(0)
		# width=100.0/crop_chr.shape[0]
		# height=int(crop_chr.shape[1]*width)
		crop_chr = cv2.resize(crop_chr,(32, 32), interpolation = cv2.INTER_AREA)
		cv2.imwrite('cropped/crop'+str(i)+'.jpg',crop_chr)
		i+=1
	# break

# cv2.imshow("Text Detection", crop_img)
# cv2.waitKey(0)


# show the output image
# cv2.imshow("Text Detection", th)

import shutil
try:  
    shutil.rmtree(path)
except OSError:  
    print ("Deletion of the directory %s failed" % path)
else:  
    print ("Successfully deleted the directory %s" % path)

"""
 python text_detection.py --image images/im2.jpg --east frozen_east_text_detection.pb

 cd Desktop\opencv-text-detection\opencv-text-detection
"""

