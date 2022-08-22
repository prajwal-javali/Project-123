from tkinter import Frame
from turtle import width
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import cv2
import PIL.ImageOps
import os,ssl,time

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image

x,y = fetch_openml("mnist_784", version = 1, return_X_y = True)
print(pd.Series(y).value_counts())
classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
nclasses = len(classes)
xtrain,xtest,ytrain,ytest = train_test_split(x,y, test_size = 2500, train_size = 7500, random_state = 5)
xtrainscaled = xtrain/255.0
xtestscaled = xtest/255.0
model = LogisticRegression(solver = "saga", multi_class = "multinomial").fit(xtrainscaled, ytrain) 
ypredict = model.predict(xtestscaled)
accuracy = accuracy_score(ytest, ypredict)
print("The Accuracy is ",accuracy) 

cap = cv2.VideoCapture(0)
while(True):
    try:
        ret,frame = cap.read()
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        height,width = gray.shape
        upperleft = (int(width/2 - 56),int(height/2 - 56))
        bottomright = (int(width/2 + 56),int(height/2 + 56))
        cv2.rectangle(gray,upperleft,bottomright,(0,255,0),2)
        roi = gray[upperleft[1]: bottomright[1],upperleft[0]: bottomright[0]]
        imPIL = Image.fromarray(roi) 
        imgbw = imPIL.convert("L")
        imgbwresized = imgbw.resize((28,28),Image.ANTIALIAS) 
        inverted = PIL.ImageOps.invert(imgbwresized)
        pixelfilter = 20
        minpixel = np.percentile(inverted,pixelfilter)
        scaled = np.clip(inverted - minpixel, 0,255)
        maxpixel = np.max(inverted) 
        scaled = np.asarray(scaled/maxpixel) 
        testsample = np.array(scaled).reshape(1,784) 
        testpredict = model.predict(testsample) 
        print("Predicted Digit is: ",testpredict) 
        cv2.imshow("frame", gray) 
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    except Exception as e:
        pass 

cap.release()
cv2.destroyAllWindows()