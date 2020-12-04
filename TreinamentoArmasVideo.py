import numpy as np
from cv2 import cv2
camera = cv2.VideoCapture(0)
car_cascade = cv2.CascadeClassifier("treinamento/cascade.xml")

while True:
    _,img = camera.read()
    height, width, c = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    objetos = car_cascade.detectMultiScale(gray, 1.2, 5)
    print(objetos)
    for (x,y,w,h) in objetos:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)

    cv2.imshow('Analise', img)
    k = cv2.waitKey(60)
    if k==27:
        break

camera.release()
cv2.destroyAllWindows()
