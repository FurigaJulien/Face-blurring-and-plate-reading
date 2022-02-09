import cv2
import numpy as np

def preprocess_frames(frame:np.ndarray,y1:int,y2:int,x1:int,x2:int,plate_color:str)->np.ndarray:
    """
    Functions to preprocess frames for pytesseract


    """
    sub_licence = frame[y1:y2, x1:x2]
    sub_licence = cv2.resize(sub_licence, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(sub_licence, cv2.COLOR_BGR2GRAY)
    invert = 255 - gray
    return invert