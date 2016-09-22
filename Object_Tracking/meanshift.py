# import the necessary packages
import numpy as np
import argparse
import cv2
import matplotlib.pyplot as plt


# construct the argument parse and parse the arguments
argument = argparse.ArgumentParser()
argument.add_argument("-v", "--input",help = "give the path to the video file")
args = vars(argument.parse_args())
print(type(args["input"]))
	
# Get the reference to the current frame
camera = cv2.VideoCapture(args["input"])

def clickpoints(event, x, y, flags, param):
	# grab the reference to the current frame, list of pts
	# points and whether or not it is pts clickpointsion mode
	global frame, boxValues, flag

	# if we are in pts clickpointsion mode, the mouse was clicked,
	# and we do not already have four points, then update the
	# list of pts points with the (x, y) location of the click
	# and draw the circle
	if flag and event == cv2.EVENT_LBUTTONDOWN and len(boxValues) < 4:
		boxValues.append((x, y))
		cv2.circle(frame, (x, y), 4, (0, 255, 0), 2)
		cv2.imshow("video", frame)

# setup the mouse click
cv2.namedWindow("video")
cv2.setMouseCallback("video", clickpoints)
		
cornerpts = None
boxValues = []
flag = False

while True:
	# grab the current frame
	(end, frame) = camera.read()

	# check for the last frame
	if not end:
		break

	if cornerpts is not None:
		# Perform mean shift algorithm on the current frame
		backprojection = cv2.calcBackProject([cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)], [0], hsvhist, [0, 180], 1)

		# Perform meanshift algorithm to the backprojection
		(r, cornerpts) = cv2.meanShift(backprojection, cornerpts, (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1))
		a,b,c,d = cornerpts
		print(a)
		print(b)
		print(c)
		print(d)
		cv2.rectangle(frame, (a,b), (a+c,b+d), 255,2)

	cv2.imshow("video", frame)
	keypress = cv2.waitKey(1) & 0xFF

	# To click points press 's'
	if keypress == ord("s") and len(boxValues) < 4:
		flag = True

		# loop untill four positions clicked
		while len(boxValues) < 4:
			cv2.imshow("video", frame)
			cv2.waitKey(0)

		# fetch the top-left and bottom-right points from the clicked points
		boxValues = np.array(boxValues)
		print(boxValues)
		print()
		pt1 = boxValues[np.argmin(boxValues.sum(axis = 1))]
		print(pt1)
		pt2 = boxValues[np.argmax(boxValues.sum(axis = 1))]
		print(pt2)

		# Fetch and save the surrounding box
		hsvhist = cv2.calcHist([cv2.cvtColor(frame.copy()[pt1[1]:pt2[1], pt1[0]:pt2[0]], cv2.COLOR_BGR2HSV)], [0], None, [5], [0, 60])
		cv2.imshow("hist", hsvhist)
		hsvhist = cv2.normalize(hsvhist, hsvhist, 0, 255, cv2.NORM_MINMAX)
		cornerpts = (pt1[0], pt1[1], pt2[0], pt2[1])

	# Exit the loop if 'e' is pressed
	elif keypress == ord("e"):
		break

camera.release()
cv2.destroyAllWindows()
