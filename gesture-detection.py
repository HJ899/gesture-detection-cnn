import cv2
from keras.models import load_model
import numpy as np

model = load_model('./gesture-model.h5')

video_capture = cv2.VideoCapture(0)
# classes_dict = {
#     'thumbs up': 0,
#     'yeet': 1,
#     'bro fist': 2,
#     'wave': 3,
#     'yo': 4,
#     'peace': 5
# }

inv_dict = {
	0: 'thumbs up',
	1: 'yeet',
	2: 'bro fist',
	3: 'wave',
	4: 'yo',
	5: 'peace'
}

x = 750 #top left x of box
y = 50  #top left y of box
h = 400 #height of box
w = 400 #width of box

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
	X = cv2.resize(img_dilation, (128,128))
	X = np.reshape(X, (1,128,128,1))
	pred = model.predict(X)
	prob = pred[0][np.argmax(pred[0])]
	text = ""
	if(prob > 0.99):
		text = inv_dict[np.argmax(pred[0])]
	font = cv2.FONT_HERSHEY_SIMPLEX
	cv2.putText(frame, text, (x+50,30), font, 1, (0, 255, 255), 2, cv2.LINE_AA)
	cv2.imshow('Video', frame)
	k = cv2.waitKey(1) & 0xFF
	if k == ord('q'):
		break 

video_capture.release()
cv2.destroyAllWindows()

