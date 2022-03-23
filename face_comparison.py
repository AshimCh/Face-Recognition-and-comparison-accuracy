import cv2
import numpy as np
import face_recognition

imgHazard = face_recognition.load_image_file('images/Hazard.jpg')
imgHazard = cv2.cvtColor(imgHazard,cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('images/hazard_test.jpg')
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgHazard)[0]
encodeHazard = face_recognition.face_encodings(imgHazard)[0]
cv2.rectangle(imgHazard,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,0),2)

faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeHazard_test = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,0),2)

results = face_recognition.compare_faces([encodeHazard],encodeHazard_test)
faceDis = face_recognition.face_distance([encodeHazard],encodeHazard_test)
print(results,faceDis)
cv2.putText(imgTest,f'{results} {round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)


cv2.imshow('Hazard',imgHazard)
cv2.imshow('Hazard Test',imgTest)

cv2.waitKey(0)