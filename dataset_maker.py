import cv2
import numpy as np
import os
import sys
video_capture = cv2.VideoCapture(0)

try:
	directory = './Images/' + sys.argv[1] 
except:
	print('\nPlease provide an argument')
	print('\nExiting...\n')
	quit()

if not os.path.exists(directory):
	os.makedirs(directory)

x = 750 #top left x of box
y = 50  #top left y of box
h = 400 #height of box
w = 400 #width of box

k = None #key press variable
b = 0 #to check if space has been pressed
count = 0
while True:
	_, frame = video_capture.read()
	frame = cv2.flip(frame,1)
	cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,255), 5)
	roi = frame[y:y+h,x:x+w] #we get the region of interest
	roi = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY) #convert it to grayscale
	_, mask = cv2.threshold(roi, 230,255,cv2.THRESH_OTSU) #use OTSU to convert grayscale to monochrome by thresholding
	mask = cv2.bitwise_not(mask) #flip the mask
	blurred = cv2.medianBlur(mask, 7) #apply median blur for smoothing of edges.
	kernel = np.ones((5,5), np.uint8)
	img_dilation = cv2.dilate(blurred, kernel, iterations=1) #apply dilation
	frame[y:y+h,x:x+w,0] = img_dilation
	frame[y:y+h,x:x+w,1] = img_dilation
	frame[y:y+h,x:x+w,2] = img_dilation
	cv2.imshow('Video', frame)
	if count<300 and b:
		path = directory + '/' + sys.argv[1] +str(count) + '.jpeg'
		cv2.imwrite(path, img_dilation)
		count+=1
	# cv2.imshow('Mask', mask)
	# cv2.imshow('blurred', blurred)
	k = cv2.waitKey(1) & 0xFF
	if k == 32:  #Press space to start capturing frames.
		b = 1
	if k == ord('q'):
		break 
video_capture.release()
cv2.destroyAllWindows()