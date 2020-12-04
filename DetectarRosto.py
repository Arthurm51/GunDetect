import cv2
import numpy as np
from datetime import datetime
import time
import os

camera = cv2.VideoCapture(0)
cascade = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")
car_cascade = cv2.CascadeClassifier("treinamento/cascade.xml")
binario = 0
while True:
    _, frame = camera.read()
    frame = cv2.flip(frame, 1)

    #YELLOW - MORPHO - BINARIZAÇÃO - BORDAS
    kernel = np.ones((5, 5), np.uint8)
    frameHsvYellow = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lowerYellow = np.array([16, 119, 129])
    upperYellow = np.array([255, 255, 255])
    mascara = cv2.inRange(frameHsvYellow, lowerYellow, upperYellow)
    resultado = cv2.bitwise_and(frame, frame, mask=mascara)
    frameGrayYellow = cv2.cvtColor(resultado, cv2.COLOR_BGR2GRAY)
    opening = cv2.morphologyEx(frameGrayYellow, cv2.MORPH_OPEN, kernel)
    _, thresh = cv2.threshold(opening, 3, 255, cv2.THRESH_BINARY)
    contornos, _ = cv2.findContours(
        thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for contorno in contornos:
        (x, y, w, h) = cv2.boundingRect(contorno)
        area = cv2.contourArea(contorno)
        if area > 1000:
            binario = 1

        else:
            binario = 0

    if binario == 1:
        frame = cv2.GaussianBlur(frame, (7, 7), 0)
        _, frame = cv2.threshold(frame, 70, 255, cv2.THRESH_BINARY_INV)
        frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frameBlur = cv2.GaussianBlur(frameGray, (7, 7), 0)

        frameAdaptativo = cv2.adaptiveThreshold(
            frameBlur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 181, 100)
        frameBordas = cv2.Canny(frameAdaptativo, 70, 150)
        objCoordenadas, objSimplificado = cv2.findContours(
        frameBordas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame, objCoordenadas, -1, (0, 0, 255), 2)
        arquivo = datetime.now().strftime('%Y%m%d_%Hh%Mm%Ss%f')+".jpg"
        timeCheck = datetime.now().strftime('%Ss')
        if os.path.isdir('armas'):
            cv2.imwrite('armas/{}'.format(arquivo),frame)
        else:
            os.makedirs('armas')
            cv2.imwrite('armas/{}'.format(arquivo),frame)

    else:
        
        #FACE
        frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detect = cascade.detectMultiScale(frameGray, 1.2, 5)
        for (x, y, w, h) in detect:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(frame, "Pessoa detectada", (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 0, cv2.LINE_AA)

        

        """
        #ARMAS
        height, width, c = frame.shape
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        objetos = car_cascade.detectMultiScale(gray, 1.2, 5)
        print(objetos)

        for (x,y,w,h) in objetos:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
            cv2.putText(frame, "Arma detectada", (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 0, cv2.LINE_AA)
            arquivo = datetime.now().strftime('%Y%m%d_%Hh%Mm%Ss%f')+".jpg"
            timeCheck = datetime.now().strftime('%Ss')
            if os.path.isdir('armas'):
             cv2.imwrite('armas/{}'.format(arquivo),frame)
            else:
                os.makedirs('armas')
                cv2.imwrite('armas/{}'.format(arquivo),frame)
        """
        



    


    

    cv2.imshow("Camera", frame)
    
    k = cv2.waitKey(60)
    if k == 27:
        break
cv2.destroyAllWindows()
camera.release()
