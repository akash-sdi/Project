import datetime
import face_recognition
import cv2 
import numpy as np 
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error as mse
from skimage.measure import compare_ssim as ssim

import math
camnum=1;

def face_distance_to_conf(face_distance, face_match_threshold=0.6):
    if face_distance > face_match_threshold:
        range = (1.0 - face_match_threshold)
        linear_val = (1.0 - face_distance) / (range * 2.0)
        return linear_val
    else:
        range = face_match_threshold
        linear_val = 1.0 - (face_distance / (range * 2.0))
        return linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))
    

acc=1
video_capture = cv2.VideoCapture(0)

aksh_image = face_recognition.load_image_file("c:/Users/SUDIPTA AKASH/OneDrive/Desktop/aksh.jpg") 
aksh_face_encoding = face_recognition.face_encodings(aksh_image)[0]


#sus_image = face_recognition.load_image_file("/home/dikshit/Files/final_app/app/known/suspect1.jpg") 
#sus_face_encoding = face_recognition.face_encodings(sus_image)[0]

#sus2_image = face_recognition.load_image_file("/home/ise/SIH/app/known/530.jpg") 
#sus2_face_encoding = face_recognition.face_encodings(sus2_image)[0]

#sir = face_recognition.load_image_file("/home/ise/app/known/sir.jpg") 
#sir_encoding = face_recognition.face_encodings(sir)[0]

#,sir_encoding

known_face_encodings = [ aksh_face_encoding ]
known_face_names = [ "vinni"] 

face_locations = [] 
face_encodings = [] 
face_names = []
process_this_frame = True
acclist=[]
while True:
    ret, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]
    if process_this_frame:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations) 
        face_names = []
        #print('this')
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding,tolerance=0.5)
            name = "Unknown"
            #print('is')
            if True in matches:
                first_match_index = matches.index(True) 
                name = known_face_names[first_match_index] 
                #acc = accuracy_score(known_face_encodings[first_match_index], face_encoding)
                #acc=mse(known_face_encodings[first_match_index], face_encoding)
                acc=ssim(known_face_encodings[first_match_index], face_encoding)
                acc=round(acc,4)
                
            face_names.append(name)
            #print('not')
    
    process_this_frame = not process_this_frame
    for (top, right, bottom, left), name in zip(face_locations, face_names): 
        top *= 4 
        right *= 4 
        bottom *= 4 
        left *= 4
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        if(name=="Unknown"):
            acclist=[1]
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            cv2.putText(frame, name , (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        else:
           
            cv2.putText(frame, name+" "+str(acc) , (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
            acclist.append(acc)
            
        with open('logfiles.txt', 'a') as the_file:
            stringto=name+" with ssim "+str(acc)+" was detected in camera "+str(camnum)+" at time "+str(datetime.datetime.now())+"\n"
            the_file.write(str(stringto))
        filename=name+".txt"
        with open(filename, 'a') as the_file:
            stringto=name+" with ssim "+str(acc)+" was detected in camera "+str(camnum)+" at time "+str(datetime.datetime.now())+"\n"
            the_file.write(str(stringto))
#rprint('okay')
            
   
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print(sum(acclist)/len(acclist))
        break

video_capture.release() 
cv2.destroyAllWindows()