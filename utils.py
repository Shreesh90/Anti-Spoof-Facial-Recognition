import cv2
import numpy as np
import dlib

######################### FACE PROFILE ###########################

detect_perfil_face = cv2.CascadeClassifier('haarcascade_profileface.xml')
detect_frontal_face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def detect(img, cascade):
    rects,_,confidence = cascade.detectMultiScale3(img, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30),
                                    flags=cv2.CASCADE_SCALE_IMAGE, outputRejectLevels = True)
    #rects = cascade.detectMultiScale(img,minNeighbors=10, scaleFactor=1.05)
    if len(rects) == 0:
        return (),()
    rects[:,2:] += rects[:,:2]
    return rects,confidence

def bounding_box(img,box,match_name=[]):
    for i in np.arange(len(box)):
        x0,y0,x1,y1 = box[i]
        img = cv2.rectangle(img,
                      (x0,y0),
                      (x1,y1),
                      (0,255,0),3);
        if not match_name:
            continue
        else:
            cv2.putText(img, match_name[i], (x0, y0-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
    return img

def convert_rightbox(img,box_right):
    res = np.array([])
    _,x_max = img.shape
    for box_ in box_right:
        box = np.copy(box_)
        box[0] = x_max-box_[2]
        box[2] = x_max-box_[0]
        if res.size == 0:
            res = np.expand_dims(box,axis=0)
        else:
            res = np.vstack((res,box))
    return res

def frontal_face(gray):
    box_frontal,w_frontal = detect(gray,detect_frontal_face)
    if len(box_frontal)==0:
	    box_frontal = []
	    name_frontal = []
    else:
 	   name_frontal = len(box_frontal)*["None"]
    return box_frontal, name_frontal

def left_face(gray):
	gray_flipped = cv2.flip(gray, 1)
	box_left, w_left = detect(gray_flipped,detect_perfil_face)
	if len(box_left)==0:
	    box_left = []
	    name_left = []
	else:
	    box_left = convert_rightbox(gray,box_left)
	    name_left = len(box_left)*["Turn Left"]
        
	return box_left, name_left

def right_face(gray):
	box_right, w_right = detect(gray,detect_perfil_face)
	if len(box_right)==0:
	    box_right = []
	    name_right = []
	else:
	    name_right = len(box_right)*["Turn Right"]
	return box_right, name_right

def face_profile(gray):
	box_frontal, name_frontal = frontal_face(gray)
	box_left, name_left = left_face(gray)
	box_right, name_right= right_face(gray)

# 	boxes = list(box_frontal)+list(box_left)+list(box_right)
	names = list(name_frontal)+list(name_left)+list(name_right)
    
	return names

#############----------- FACE PROFILE --------------##############



######################## EYE BLINKING #############################


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

def midpt(p1, p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

def length(diff1, diff2):
    return np.sqrt( diff1**2 + diff2**2)

def eye_blink_ratio(landmarks, points):
    leftpt  = (landmarks.part(points[0]).x, landmarks.part(points[0]).y)
    rightpt = (landmarks.part(points[1]).x, landmarks.part(points[1]).y)
    top_left = (landmarks.part(points[2]).x, landmarks.part(points[2]).y)
    top_right = (landmarks.part(points[3]).x, landmarks.part(points[3]).y)
    bottom_left = (landmarks.part(points[5]).x, landmarks.part(points[5]).y)
    bottom_right = (landmarks.part(points[4]).x, landmarks.part(points[4]).y)
    
    # centre_top = midpt(landmarks.part(points[2]), landmarks.part(points[3]))
    # centre_bottom = midpt(landmarks.part(points[4]), landmarks.part(points[5]))
    
    # cv2.circle(frame, leftpt, 2, (0,0,255), -1)
    # cv2.circle(frame, rightpt, 2, (0,0,255), -1)
    # cv2.circle(frame, top_left, 2, (0,0,255), -1)
    # cv2.circle(frame, top_right, 2, (0,0,255), -1)
    # cv2.circle(frame, bottom_left, 2, (0,0,255), -1)
    # cv2.circle(frame, bottom_right, 2, (0,0,255), -1)
    
    # hor_line = cv2.line(frame, leftpt, rightpt, (0,255,0), 1)
    # ver_line = cv2.line(frame, centre_top, centre_bottom, (0,255,0), 1)

    hor_len = length(leftpt[0]-rightpt[0], leftpt[1]-rightpt[1])
    ver_len1 = length(top_left[0]-bottom_left[0], top_left[1]-bottom_left[1])
    ver_len2 = length(top_right[0]-bottom_right[0], top_right[1]-bottom_right[1])
    
    ear = (ver_len1 + ver_len2)/(2 * hor_len)
    
    # ratio = hor_len/ver_len
    return ear




def eye_blinking(imgray):
    faces = detector(imgray)
    
    both_frames=0
    left_frames=0
    right_frames=0

    profile = ""   
    
    for face in faces:
        # x1 = face.left()
        # y1 = face.top()
        # x2 = face.right()
        # y2 = face.bottom()
        
        # cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 2)
        landmarks = predictor(imgray, face)
        # cv2.circle(frame, (landmarks.part(36).x, landmarks.part(36).y), 2, (0,0,255), -1)
        
        ratio_right = eye_blink_ratio(landmarks, [36, 39, 37, 38, 40,41])
        ratio_left = eye_blink_ratio(landmarks, [42, 45, 43, 44, 46,47])
        
        ratio = (ratio_left + ratio_right)/2

        # print(ratio_left, "       ", ratio_right)
        # print(ratio)
        if ratio_left <= 0.17 and ratio_right <= 0.17:
            both_frames += 1
            if both_frames==1:
                # cv2.putText(frame, 'BLINKED',(50, 50),  cv2.FONT_HERSHEY_DUPLEX , 2, (0,0,255), 10)
                both_frames=0
                profile = "Blink both eyes"
       
        elif ratio_left < 0.24 and ratio_right >= 0.25:
            left_frames += 1
            if left_frames==1:
                # cv2.putText(frame, 'LEFT BLINKED',(50, 50),  cv2.FONT_HERSHEY_DUPLEX, 2, (0,0,255), 10)
                left_frames=0
                profile = "Blink Left eye"
        elif ratio_right < 0.24 and ratio_left >= 0.25:
            right_frames += 1
            if right_frames==1:
                # cv2.putText(frame, 'RIGHT BLINKED',(50, 50),  cv2.FONT_HERSHEY_DUPLEX, 2, (0,0,255), 10)
                right_frames=0
                profile = "Blink Right eye"
        else:
            profile = "None"
	
        return profile
        
##############------------- EYE BLINKING -----------#################
 

##################### EMOTION DETECTION #########################

from keras.models import load_model
from keras.preprocessing.image import img_to_array

classifier = load_model('Emotion_little_vgg5_2.h5')
class_labels = ['Angry Face', 'Happy Face', 'None', 'Sad Face','Surprise Face']

def face_emotion(imgray):
    faces = detect_frontal_face.detectMultiScale(imgray, 1.3, 5)
    
    for x,y,w,h in faces:
        # cv2.rectangle(frame, (x, y), (x+w, y+h), (0,0,255), 2)
        roi_gray = imgray[y:y+h, x:x+h]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
        
        
        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            
            preds = classifier.predict(roi)[0]
            label = class_labels[preds.argmax()]
            emotion = label
            # return label
        else:
            emotion = "None"
            # return "None"
        
        return emotion

##########------------- EMOTION DETECTION ----------------##############
    
    
##################### FACE RECOGNITION ##################

from face_recognition import api
import os

path = 'known_faces'
images = []
known_names = []

Names = os.listdir(path)
# print(Names)

for name in Names:
    currImage = cv2.imread(f'{path}/{name}') 
    images.append(currImage)
    known_names.append(os.path.splitext(name)[0])
    
# print(known_names)

def findEncodings(images):
    encodingList = []
    for image in images:
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # FACE RECOGNITION TAKES RGB IMAGES
        encoding = api.face_encodings(img)[0]
        encodingList.append(encoding)
    return encodingList

KnownEncodingList = findEncodings(images)

def face_rec(frame):
    smallimg = cv2.resize(frame, (0,0), None, 0.25,0.25)
    smallimg = cv2.cvtColor(smallimg, cv2.COLOR_BGR2RGB)
    
    facescurrFrame = api.face_locations(smallimg)
    encodecurrFrame = api.face_encodings(smallimg, facescurrFrame)
    
    for encodeFace,faceLoc in zip(encodecurrFrame, facescurrFrame):
        matches = api.compare_faces(KnownEncodingList, encodeFace)        
        faceDist = api.face_distance(KnownEncodingList, encodeFace)
        matchIndex = np.argmin(faceDist)
        
        if matches[matchIndex]:
            name = known_names[matchIndex].upper()
            
        return name
    
    
#########---------------- FACE RECOGNITION------------###########
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
