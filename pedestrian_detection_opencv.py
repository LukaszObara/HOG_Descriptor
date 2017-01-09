# Libraries
# Standard Libraries
import os

# Third-Party Libraries
import cv2
import imutils
from imutils.object_detection import non_max_suppression
import numpy as np

def people_detect(file_path):

	hog = cv2.HOGDescriptor()
	hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
	
	im = cv2.imread(file_path)
	im = imutils.resize(im, width=min(360, im.shape[1]))
	box = 0

	# detect people in the image
	(rects, weights) = hog.detectMultiScale(im, winStride=(4, 4),
											padding=(4, 4), scale=1.1)

	if np.any(rects) > 0:
		box += 1

	# draw the original bounding boxes
	for (x, y, w, h) in rects:
		cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 2)

	# apply non-maxima suppression to the bounding boxes 
	rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
	pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

	# draw the final bounding boxes
	for (xA, yA, xB, yB) in pick:
		cv2.rectangle(im, (xA, yA), (xB, yB), (0, 255, 0), 2)

	# show the output images
	cv2.imshow("People Detector", im)
	cv2.waitKey(0)

def main():
	location = 'C:\\Users\\lukas\\OneDrive\\Documents\\Machine Learning\\' \
		 	 + 'Pedestrian_detection\\Datasets\\INRIA\\Train\\pos\\'
	files = os.listdir(location)

	file_path = location + str(files[-4])

	people_detect(file_path)

if __name__ == '__main__':
	pass
