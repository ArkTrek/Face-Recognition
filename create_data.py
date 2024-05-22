# Creating database 
# It captures images and stores them in datasets 
# folder under the folder name of sub_data 
import cv2, sys, numpy, os 
haar_file = 'haarcascade_frontalface_default.xml'

# All the faces data will be 
# present this folder 
datasets = 'dataset'
sub_data = input('Enter Folder Name[Face]:')

# These are sub data sets of folder, 
# for my faces I've used my name you can 
# change the label here 


path = os.path.join(datasets, sub_data) 
if not os.path.isdir(path): 
	os.mkdir(path) 

# defining the size of images 
(width, height) = (810, 700)	 

#'0' is used for my webcam, 
# if you've any other camera 
# attached use '1' like this 
face_cascade = cv2.CascadeClassifier(haar_file) 
webcam = cv2.VideoCapture(0) 

# The program loops until it has 11 images of the face. 
count = 0
i = 0
imageCount = int(input("Enter number of image captures: "))
print("File Location: ",datasets,"/",sub_data)
while count < imageCount:
	count += 1
	(_, im) = webcam.read() 
	gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) 
	faces = face_cascade.detectMultiScale(gray, 1.3, 4) 
	for (x, y, w, h) in faces: 
		cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2) 
		face = gray[y:y + h, x:x + w] 
		face_resize = cv2.resize(face, (width, height)) 
		cv2.imwrite('% s/% s.png' % (path, count + 1), face_resize)
		print("Image Captured " ,i+1,"/",imageCount)
		i +=1
	
	cv2.imshow('OpenCV', im)
	
	# Waiting Time - in Sec
	key = cv2.waitKey(30) 
	if key == 27: 
		break