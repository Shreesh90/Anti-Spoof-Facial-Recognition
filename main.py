import cv2
import dlib
import numpy as np
import utils
# from time import sleep
# import sys


detect_perfil_face = cv2.CascadeClassifier('haarcascade_profileface.xml')
detect_frontal_face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


activity_functions = [ utils.face_profile, utils.eye_blinking ]
activities = ["Turn Left", "Turn Right", "Blink both eyes", "Blink Left eye", "Blink Right eye", "Angry Face", "Happy Face", "Surprise Face"]

profile_status = {"None" : False,
                  "Turn Left": False,
                  "Turn Right": False,}

eye_status = {"None": False,
              "Blink both eyes": False,
              "Blink Left eye": False,
              "Blink Right eye": False}

emotion_status = {"None": False,
                  "Happy Face": False,
                  "Sad Face": False,
                  "Surprise Face": False,
                  "Angry Face": False}


def face_state(status, gray):
    
    ##### PROFILE #####
    if status == "profile":
        curr_state = profile_status.copy()
      
        orientation = utils.face_profile(gray)
        if not orientation or orientation[0] == "None":
            curr_state["None"] = False
        # elif orientation is None:
        #     curr_state[f"{orientation}"] = False
        else:
            curr_state[orientation[0]] = True 
        
    ##### EYE ######
    elif status == "eye":
        curr_state = eye_status.copy()
        
        blink = utils.eye_blinking(gray)
        if blink == "None":
            curr_state[blink] = False
        elif blink == None:
            curr_state[f"{blink}"] = False
        else:
            curr_state[blink] = True
            
    ##### EMOTION #####
    elif status == "emotion":
        curr_state = emotion_status.copy()
        
        emotion = utils.face_emotion(gray)
        if emotion == "None" or emotion == "Sad Face":
            curr_state[emotion] = False
        elif emotion == None:
            curr_state[f"{emotion}"] = False
        else:
            curr_state[emotion] = True

    return curr_state    

def false_check(curr_state):
    
    for activity in curr_state:
            if activity == task:
                access = "granted"
                continue
            if curr_state[activity] == True:
                access = "denied"
                return access 

divert = False

indices = np.random.randint(8, size=4)
# indices = [6,1,7,2]
print(indices)
# indices = [0] 
task_count=0

cap = cv2.VideoCapture(0)
for index in indices:
    task = activities[index]
    
    if index <= 1:
        status = "profile"
    elif index >= 2 and index <= 4:
        status = "eye"
    else:
        status = "emotion"
        
    false_check_frames = 0
    truth_check_frames = 0
    
    while cap.isOpened():
        _, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        curr_state = face_state(status, gray)
        cv2.putText(frame, task, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 3)
        
        print(curr_state)
        
        value = curr_state[task]
        if value is True:
            cv2.putText(frame, f": {value}", (300,50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 3)
        elif value is False:
            cv2.putText(frame, f": {value}", (300,50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 3)
        
        cv2.imshow('LIVE',frame)
        cv2.waitKey(1)
        if value is True:
            truth_check_frames += 1
            false_check_frames = 0
            if truth_check_frames == 8:
                task_count += 1
                if task_count == 4:
                    
                    name = utils.face_rec(frame)
                    print(name)
                    
                    cv2.putText(frame, f"Welcome {name}!", (10,300), cv2.FONT_HERSHEY_SIMPLEX , 1, (0,255,255), 3  , cv2.LINE_AA)
                    cv2.imshow('LIVE', frame)
                    
                    
                    
                    cv2.waitKey(1000)
                break
            
        else:
            access = false_check(curr_state)
            if access == "denied":
                false_check_frames += 1
                truth_check_frames = 0
                if false_check_frames == 10:
                    
                    name = utils.face_rec(frame)
                    print(name)
                    
                    cv2.putText(frame, f"You are not {name}! Access Denied.", (10,300), cv2.FONT_HERSHEY_SIMPLEX , 1, (0,0,255), 3  , cv2.LINE_AA)
                    cv2.imshow('LIVE', frame)
                
                    cv2.waitKey(3000)
                    divert = True
                    break
            continue
    if divert == True:
        break
       
        
cap.release()
cv2.destroyAllWindows()









